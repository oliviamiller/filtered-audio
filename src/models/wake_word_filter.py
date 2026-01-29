from typing import (
    ClassVar,
    Mapping,
    Sequence,
    Tuple,
    cast,
    List,
    AsyncGenerator,
    Any,
    Optional,
)
import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from vosk import Model as VoskModel, KaldiRecognizer
import webrtcvad
from typing_extensions import Self

from .fuzzy_matcher import FuzzyWakeWordMatcher
from viam.components.audio_in import AudioIn, AudioResponse as AudioChunk
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict
from viam.streams import StreamWithIterator

from .vosk import get_vosk_model, DEFAULT_VOSK_MODEL

# Default configuration values
DEFAULT_VAD_AGGRESSIVENESS = 3  # 0-3, higher = less sensitive
AUDIO_SAMPLE_RATE_HZ = 16000
MAX_BUFFER_SIZE_BYTES = 480000  # ~15 seconds at 16kHz
DEFAULT_SILENCE_DURATION_MS = 900  # milliseconds of silence before ending a speech segment
DEFAULT_MIN_SPEECH_DURATION_MS = 300  # min length of speech to process
DEFAULT_GRAMMAR_CONFIDENCE = 0.7  # min confidence for Vosk grammar matches (0.0-1.0)

class WakeWordFilter(AudioIn, EasyResource):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("viam", "filtered-audio"), "wake-word-filter"
    )

    # Instance variables
    logger: logging.Logger
    wake_words: List[str]
    vad: webrtcvad.Vad
    vosk_model: VoskModel
    executor: ThreadPoolExecutor
    is_shutting_down: bool
    microphone_client: AudioIn
    fuzzy_matcher: Optional[FuzzyWakeWordMatcher]
    silence_duration_ms: int
    min_speech_duration_ms:  int
    detection_paused: bool
    grammar_confidence: float
    use_grammar: bool

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        instance = super().new(config, dependencies)

        attrs = struct_to_dict(config.attributes)
        microphone = str(attrs.get("source_microphone", ""))
        wake_words = attrs.get("wake_words", [])

        # Handle both single string and list
        if isinstance(wake_words, str):
            instance.wake_words = [wake_words.lower()] if wake_words else []
        elif isinstance(wake_words, list):
            instance.wake_words = [str(w).lower() for w in wake_words]
        else:
            instance.wake_words = []

        if instance.wake_words:
            instance.logger.info("Wake words: %s", instance.wake_words)
        else:
            instance.logger.info("No wake words configured - all speech will be sent")

        model = str(attrs.get("vosk_model", DEFAULT_VOSK_MODEL))
        vad_aggressiveness = int(
            attrs.get("vad_aggressiveness", DEFAULT_VAD_AGGRESSIVENESS)
        )

        # Fuzzy matching - enabled if fuzzy_threshold is set
        fuzzy_threshold = attrs.get("fuzzy_threshold", None)
        if fuzzy_threshold is not None:
            instance.fuzzy_matcher = FuzzyWakeWordMatcher(
                threshold=int(fuzzy_threshold)
            )
            instance.logger.info(
                f"Fuzzy matching enabled with threshold={fuzzy_threshold}"
            )
        else:
            instance.fuzzy_matcher = None

        instance.silence_duration_ms = int(
            attrs.get("silence_duration_ms", DEFAULT_SILENCE_DURATION_MS)
        )
        instance.logger.info(
            f"VAD Silence duration: {instance.silence_duration_ms}ms"
        )

        instance.min_speech_duration_ms = int(attrs.get("min_speech_ms", DEFAULT_MIN_SPEECH_DURATION_MS))
        instance.logger.info(
            f"min speech segment duration: {instance.min_speech_duration_ms}ms"
        )

        # Grammar mode - default True for constrained wake word recognition
        instance.use_grammar = attrs.get("use_grammar", True)
        instance.logger.info(
            f"Vosk grammar mode: {instance.use_grammar}"
        )

        instance.grammar_confidence = float(
            attrs.get("vosk_grammar_confidence", DEFAULT_GRAMMAR_CONFIDENCE)
        )
        instance.logger.info(
            "Vosk grammar confidence threshold: %.2f", instance.grammar_confidence
        )

        # Initialize WebRTC VAD
        instance.vad = webrtcvad.Vad(vad_aggressiveness)
        instance.logger.info(
            f"WebRTC VAD initialized with aggressiveness: {vad_aggressiveness}"
        )

        # Load Vosk model (checks bundled, then cached, then downloads)
        model_path = get_vosk_model(model, instance.logger)
        instance.vosk_model = VoskModel(model_path)
        instance.logger.debug("Vosk model loaded")

        # Create thread pool for Vosk processing (non-blocking)
        instance.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vosk")
        instance.is_shutting_down = False
        instance.logger.debug("Thread pool executor created for Vosk processing")

        # Get source microphone
        if microphone:
            mic = dependencies[AudioIn.get_resource_name(microphone)]
            instance.microphone_client = cast(AudioIn, mic)
        else:
            instance.logger.error(
                "wake-word-filter must have a source microphone, no microphone found"
            )
            raise RuntimeError(
                "wake-word-filter must have a source microphone, no microphone found"
            )

        # Detection pause state (for muting during TTS playback)
        instance.detection_paused = False

        return instance

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        deps: List[str] = []
        attrs = struct_to_dict(config.attributes)
        mic: Any = attrs.get("source_microphone", "")

        if mic == "":
            raise ValueError("source_microphone attribute is required")
        if not isinstance(mic, str):
            raise ValueError("source_microphone attribute must be a string")
        deps.append(mic)

        # Validate wake_words (optional - if empty, all speech is sent)
        wake_words: Any = attrs.get("wake_words", [])
        if isinstance(wake_words, str):
            # Single string is valid
            pass
        elif isinstance(wake_words, list):
            # List must contain only strings
            for word in wake_words:
                if not isinstance(word, str):
                    raise ValueError(
                        f"All wake_words must be strings, got {type(word).__name__}"
                    )
        else:
            raise ValueError(
                f"wake_words must be a string or list of strings, got {type(wake_words).__name__}"
            )

        # Validate VAD aggressiveness
        vad_aggressiveness: Any = attrs.get("vad_aggressiveness", None)
        if vad_aggressiveness is not None:
            if (
                not isinstance(vad_aggressiveness, (int, float))
                or vad_aggressiveness % 1 != 0
            ):
                raise ValueError("vad_aggressiveness must be a whole number")
        if vad_aggressiveness is not None and not 0 <= vad_aggressiveness <= 3:
            raise ValueError(
                f"vad_aggressiveness must be 0-3, got {vad_aggressiveness}"
            )

        # Validate fuzzy threshold
        fuzzy_threshold: Any = attrs.get("fuzzy_threshold", None)
        if fuzzy_threshold is not None:
            if (
                not isinstance(fuzzy_threshold, (int, float))
                or fuzzy_threshold % 1 != 0
            ):
                raise ValueError("fuzzy_threshold must be a whole number")
        if fuzzy_threshold is not None and not 0 <= fuzzy_threshold <= 5:
            raise ValueError(f"fuzzy_threshold must be 0-5, got {fuzzy_threshold}")

        # Validate silence_duration_ms
        silence_duration_ms: Any = attrs.get("silence_duration_ms", None)
        if silence_duration_ms is not None:
            if (
                not isinstance(silence_duration_ms, (int, float))
                or silence_duration_ms % 1 != 0
            ):
                raise ValueError("silence_duration_ms must be a whole number")
            if silence_duration_ms <= 0:
                raise ValueError("silence_duration_ms must be positive")

        # Validate min_speech_ms
        min_speech_ms: Any = attrs.get("min_speech_ms", None)
        if min_speech_ms is not None:
            if (
                not isinstance(min_speech_ms, (int, float))
                or min_speech_ms % 1 != 0
            ):
                raise ValueError("min_speech_ms must be a whole number")
            if min_speech_ms <= 0:
                raise ValueError("min_speech_ms must be positive")

        # Validate use_grammar
        use_grammar: Any = attrs.get("use_grammar", None)
        if use_grammar is not None:
            if not isinstance(use_grammar, bool):
                raise ValueError("use_grammar must be a boolean")

        # Validate grammar_confidence
        grammar_confidence: Any = attrs.get("vosk_grammar_confidence", None)
        if grammar_confidence is not None:
            if not isinstance(grammar_confidence, (int, float)):
                raise ValueError("vosk_grammar_confidence must be a number")
            if grammar_confidence < 0.0 or grammar_confidence > 1.0:
                raise ValueError(
                    f"vosk_grammar_confidence must be 0.0-1.0, got {grammar_confidence}"
                )

        return deps, []

    async def _process_speech_segment(
        self, speech_chunk_buffer: List[AudioChunk], speech_buffer: bytearray
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Check buffered audio for wake words and yield chunks if detected.

        Args:
            speech_chunk_buffer: List of audio cqhunks to yield if wake word found
            speech_buffer: Accumulated speech audio bytes to process with Vosk
        """
        if not speech_chunk_buffer:
            return

        # Don't process if we're shutting down
        if self.is_shutting_down:
            self.logger.debug("Skipping speech processing due to shutdown")
            return

        try:
            # If no wake words configured, send all speech
            if not self.wake_words:
                self.logger.debug(
                    "No wake words configured, yielding %d chunks (%d bytes)",
                    len(speech_chunk_buffer), len(speech_buffer)
                )
                for chunk in speech_chunk_buffer:
                    yield chunk
                empty_response = AudioChunk()
                empty_response.audio.audio_data = b""
                yield empty_response
                return

            wake_word_detected = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                self._check_for_wake_word,
                bytes(speech_buffer),
                AUDIO_SAMPLE_RATE_HZ,
            )

            if wake_word_detected:
                self.logger.info(
                    "Wake word detected! Yielding %d chunks (%d bytes)",
                    len(speech_chunk_buffer), len(speech_buffer)
                )
                for chunk in speech_chunk_buffer:
                    yield chunk
                # Yield empty chunk to signal segment end
                empty_response = AudioChunk()
                empty_response.audio.audio_data = b""
                yield empty_response
                self.logger.debug("Sent empty chunk to signal segment end")
        except RuntimeError as e:
            if "shutdown" in str(e).lower():
                self.logger.debug("Executor shutdown during processing, ignoring")
                return
            raise

    async def get_audio(
        self, codec: str, duration_seconds: float, previous_timestamp_ns: int, **kwargs
    ) -> StreamWithIterator:
        """
        Stream audio, yielding buffered audio chunks when a wake word detected.

        Uses WebRTC VAD to detect speech, then Vosk to find wake words.

        Args:
            codec: Audio codec (should be "pcm16")
            duration_seconds: Duration (use 0 for continuous)
            previous_timestamp_ns: Previous timestamp (use 0 to start from now)

        Yields:
            AudioResponse chunks
        """
        if not self.microphone_client:
            raise ValueError("no source microphone found")

        if codec.lower() != "pcm16":
            self.logger.error(f"Unsupported codec: {codec}. Only PCM16 is supported.")
            raise ValueError(
                f"Wake word filter only supports PCM16 codec, got: {codec}"
            )

        async def audio_generator() -> AsyncGenerator[AudioChunk, None]:
            self.logger.info(
                f"Starting speech detection with VAD... (duration_seconds={duration_seconds})"
            )

            # Check mic properties
            mic_props = await self.microphone_client.get_properties()
            self.logger.debug(
                f"Microphone properties - Sample rate: {mic_props.sample_rate_hz} Hz, Channels: {mic_props.num_channels}"
            )

            # Validate sample rate (Vosk models are trained on 16000 Hz)
            if mic_props.sample_rate_hz != AUDIO_SAMPLE_RATE_HZ:
                raise ValueError(
                    f"Wake word filter requires 16000 Hz audio, "
                    f"but source microphone provides {mic_props.sample_rate_hz} Hz. "
                    f"Please configure source microphone to output 16000 Hz PCM16 audio."
                )

            # Validate mono audio (Vosk requires single channel)
            if mic_props.num_channels != 1:
                raise ValueError(
                    f"Wake word filter requires mono (1 channel) audio, "
                    f"but source microphone provides {mic_props.num_channels} channels. "
                    f"Please configure source microphone for mono audio."
                )

            mic_stream = await self.microphone_client.get_audio(
                codec, duration_seconds, previous_timestamp_ns
            )
            self.logger.info(
                f"Microphone stream started (requested duration: {duration_seconds}s)"
            )

            speech_chunk_buffer: list[
                AudioChunk
            ] = []  # Audio chunks that contain speech
            speech_buffer = (
                bytearray()
            )  # Accumulates speech frames (raw audio bytes) for Vosk
            audio_buffer = (
                bytearray()
            )  # Working buffer for breaking chunks into VAD frames

            is_speech_active = False
            silence_frames = 0
            speech_frames = 0  # Track how much speech we've heard
            frame_duration_ms = 30
            max_silence_frames = self.silence_duration_ms // frame_duration_ms
            min_speech_frames = self.min_speech_duration_ms // frame_duration_ms

            def reset_buffers():
                """Clear all buffers and reset speech state."""
                nonlocal is_speech_active, silence_frames, speech_frames
                speech_chunk_buffer.clear()
                speech_buffer.clear()
                audio_buffer.clear()
                is_speech_active = False
                silence_frames = 0
                speech_frames = 0

            async for audio_chunk in mic_stream:
                # Exit stream if shutting down
                if self.is_shutting_down:
                    self.logger.info("Stream ending due to shutdown")
                    break

                # Skip processing when detection is paused (e.g., during TTS)
                if self.detection_paused:
                    # Clear any buffered speech to avoid stale data
                    if speech_chunk_buffer or speech_buffer:
                        reset_buffers()
                    continue

                audio_data = audio_chunk.audio.audio_data

                if not audio_data:
                    continue

                # WebRTC VAD requires specific frame sizes (10, 20, or 30ms)
                # At 16kHz: 30ms = 480 samples = 960 bytes
                frame_size = 960
                should_process = False

                # Track if we've added this chunk to the buffer yet
                chunk_added = False

                # Add new audio data to working buffer
                audio_buffer.extend(audio_data)

                # Process complete 30ms frames
                while len(audio_buffer) >= frame_size:
                    frame = audio_buffer[:frame_size]

                    # Check if frame contains speech
                    try:
                        # Note: WebRTC VAD only accepts 16-bit mono PCM audio,
                        # sampled at 8000, 16000, or 32000 Hz.
                        # A frame must be either 10, 20, or 30 ms in duration
                        is_speech = self.vad.is_speech(frame, mic_props.sample_rate_hz)
                    except Exception as e:
                        self.logger.error(f"VAD error: {e}")
                        is_speech = False

                    if is_speech:
                        if not is_speech_active:
                            self.logger.debug("Speech segment started")
                            is_speech_active = True
                        silence_frames = 0
                        speech_frames += 1

                        # Buffer this speech frame
                        if not chunk_added:
                            speech_chunk_buffer.append(audio_chunk)
                            chunk_added = True
                        speech_buffer.extend(frame)
                    else:
                        if is_speech_active:
                            # Buffer silence frames during active speech segment
                            if not chunk_added:
                                speech_chunk_buffer.append(audio_chunk)
                                chunk_added = True
                            speech_buffer.extend(frame)

                            silence_frames += 1

                            if silence_frames >= max_silence_frames:
                                should_process = True
                                audio_buffer = audio_buffer[
                                    frame_size:
                                ]  # Remove processed frame
                                break

                    # Remove the frame we just processed
                    audio_buffer = audio_buffer[frame_size:]

                # If speech segment ended, check for wake word
                if should_process:
                    # Only process if we had enough speech (filters out brief false positives)
                    if speech_frames >= min_speech_frames:
                        self.logger.debug(
                            f"Speech segment ended ({speech_frames} frames), checking for wake word"
                        )
                        async for chunk in self._process_speech_segment(
                            speech_chunk_buffer, speech_buffer
                        ):
                            yield chunk
                    else:
                        self.logger.debug(
                            f"Ignoring false positive: only {speech_frames} frames detected"
                        )

                    reset_buffers()

                # Prevent buffer from growing too large, process when it gets to max size
                if len(speech_buffer) > MAX_BUFFER_SIZE_BYTES:
                    self.logger.debug("Processing speech segment")
                    async for chunk in self._process_speech_segment(
                        speech_chunk_buffer, speech_buffer
                    ):
                        yield chunk

                    reset_buffers()

            # Process any remaining buffered audio when stream ends
            if speech_chunk_buffer and speech_frames >= min_speech_frames:
                self.logger.debug(
                    f"Stream ended with {len(speech_buffer)} bytes buffered, processing"
                )
                async for chunk in self._process_speech_segment(
                    speech_chunk_buffer, speech_buffer
                ):
                    yield chunk
            elif speech_chunk_buffer:
                self.logger.debug(
                    f"Stream ended: ignoring buffered audio (only {speech_frames} frames, likely false positive)"
                )

        return StreamWithIterator(audio_generator())

    def _check_for_wake_word(self, audio_bytes: bytes, sample_rate: int) -> bool:
        """
        Check if any wake word is in audio.

        Args:
            audio_bytes: Raw PCM16 audio data
            sample_rate: Audio sample rate

        Returns:
            bool: True if any wake word detected
        """
        try:
            if self.use_grammar:
                # Grammar mode: constrain recognition to only wake words
                grammar = json.dumps(self.wake_words)
                recognizer = KaldiRecognizer(self.vosk_model, sample_rate, grammar)
            else:
                # Full transcription mode: recognize all speech
                recognizer = KaldiRecognizer(self.vosk_model, sample_rate)

            recognizer.SetWords(True)  # Enable word-level confidence scores
            recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(recognizer.FinalResult())
            text = result.get("text", "").lower()

            # Check confidence to reduce false positives from grammar forcing
            self.logger.debug("Vosk result: %s", result)
            if "result" in result and result["result"]:
                avg_conf = sum(w.get("conf", 1.0) for w in result["result"])
                avg_conf /= len(result["result"])
                self.logger.debug("Vosk confidence: %.2f", avg_conf)
                if avg_conf < self.grammar_confidence:
                    self.logger.debug(
                        "Rejecting low confidence: '%s' (conf=%.2f < %.2f)",
                        text, avg_conf, self.grammar_confidence
                    )
                    return False

            if text:
                self.logger.debug(f"Recognized: '{text}'")
            else:
                self.logger.debug("Vosk returned empty text, no speech recognized")
                return False

            for wake_word in self.wake_words:
                if self.fuzzy_matcher and not self.use_grammar:
                    match_details = self.fuzzy_matcher.match(text, wake_word)
                    if match_details:
                        self.logger.info(
                            f"Wake word '{wake_word}' detected (fuzzy: "
                            f"'{match_details['matched_text']}', "
                            f"distance={match_details['distance']})"
                        )
                        return True
                else:
                    pattern = rf"\b{re.escape(wake_word)}\b"
                    match = re.search(pattern, text)
                    self.logger.debug(
                        "Checking wake_word='%s' pattern='%s' against text='%s' -> %s",
                        wake_word, pattern, text, match
                    )
                    if match:
                        self.logger.info("Wake word '%s' detected", wake_word)
                        return True

            self.logger.debug("No wake word match found")
            return False
        except Exception as e:
            self.logger.error(f"Vosk error: {e}", exc_info=True)
            return False

    async def close(self) -> None:
        # Signal shutdown to prevent new tasks
        self.is_shutting_down = True

        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
            self.logger.debug("Thread pool executor shut down")

    async def do_command(
        self, command: Mapping[str, Any], **kwargs
    ) -> Mapping[str, Any]:
        """
        Handle commands for the wake word filter.

        Supported commands:
        - pause_detection: Pause wake word detection
        - resume_detection: Resume wake word detection

        Examples:
            {"pause_detection": None}  # pause detection
            {"resume_detection": None}    # resume detection
        """
        if "pause_detection" in command:
            self.detection_paused = True
            self.logger.info("Detection paused")
            return {"status": "paused"}

        elif "resume_detection" in command:
            self.detection_paused = False
            self.logger.info("Detection resumed")
            return {"status": "resumed"}

        else:
            raise ValueError(f"Unknown command keys: {list(command.keys())}")

    async def get_geometries(self, **kwargs) -> List[Any]:
        raise NotImplementedError()

    async def get_properties(self, **kwargs) -> AudioIn.Properties:
        # Return properties from underlying microphone
        props = await self.microphone_client.get_properties()
        # Ensure we only report PCM16 support
        return AudioIn.Properties(
            supported_codecs=["pcm16"],
            sample_rate_hz=props.sample_rate_hz,
            num_channels=props.num_channels,
        )
