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
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from vosk import Model as VoskModel, KaldiRecognizer
import webrtcvad
import numpy as np
from openwakeword.model import Model as OWWModel
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
DEFAULT_SILENCE_DURATION_MS = 900  # milliseconds of silence before ending a speech segment
AUDIO_SAMPLE_RATE_HZ = 16000
MAX_BUFFER_SIZE_BYTES = 480000  # ~15 seconds at 16kHz
DEFAULT_MIN_SPEECH_MS = 300


def _get_bundled_oww_models_dir() -> str:
    """Get the path to bundled OpenWakeWord models, handling PyInstaller bundles."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_path = Path(sys._MEIPASS)
    else:
        # Running in development
        base_path = Path(__file__).resolve().parents[2]
    return str(base_path / "oww_models")


# OpenWakeWord model download URLs
OWW_MODEL_URLS = {
    "alexa_v0.1": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.tflite",
    "hey_jarvis_v0.1": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.tflite",
    "hey_mycroft_v0.1": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_mycroft_v0.1.tflite",
    "hey_rhasspy_v0.1": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_rhasspy_v0.1.tflite",
}

# Base models required by OpenWakeWord
OWW_BASE_MODEL_URLS = {
    "melspectrogram": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite",
    "embedding_model": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite",
}


def _download_file(url: str, dest_path: str, logger: logging.Logger) -> None:
    """Download a file from URL to destination path."""
    import ssl
    import urllib.request
    import certifi

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(url, context=ssl_context, timeout=120) as response:
            with open(dest_path, "wb") as out_file:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    out_file.write(chunk)
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise RuntimeError(f"Failed to download {url}: {e}")


def _setup_oww_models(model_name: str, logger: logging.Logger) -> str:
    """
    Setup OpenWakeWord models, copying/downloading to openwakeword's expected location.

    Returns:
        Path to the wakeword model file
    """
    import shutil
    import openwakeword

    bundled_dir = _get_bundled_oww_models_dir()
    data_path = os.getenv("VIAM_MODULE_DATA")

    # Get openwakeword's expected model directory
    oww_pkg_dir = os.path.dirname(openwakeword.__file__)
    oww_models_dir = os.path.join(oww_pkg_dir, "resources", "models")
    os.makedirs(oww_models_dir, exist_ok=True)

    def ensure_model(name: str, urls: dict, is_base: bool = False) -> str:
        """Ensure model exists in openwakeword's expected location."""
        # Determine filename
        if name in urls:
            filename = urls[name].split("/")[-1]
        else:
            filename = f"{name}.tflite"

        target_path = os.path.join(oww_models_dir, filename)

        # Already in place?
        if os.path.exists(target_path):
            logger.debug(f"Model already exists: {target_path}")
            return target_path

        # Check bundled
        for ext in [".tflite", ".onnx"]:
            bundled_path = os.path.join(bundled_dir, f"{name}{ext}")
            if os.path.exists(bundled_path):
                logger.info(f"Copying bundled model to {target_path}")
                shutil.copy2(bundled_path, target_path)
                return target_path

        # Check VIAM_MODULE_DATA cache
        if data_path:
            for ext in [".tflite", ".onnx"]:
                cached = os.path.join(data_path, f"{name}{ext}")
                if os.path.exists(cached):
                    logger.info(f"Copying cached model to {target_path}")
                    shutil.copy2(cached, target_path)
                    return target_path

        # Download
        if name not in urls:
            raise RuntimeError(
                f"Unknown model: {name}. Available: {list(urls.keys())}"
            )

        url = urls[name]
        logger.info(f"Downloading {name} from {url}...")
        _download_file(url, target_path, logger)
        return target_path

    # Setup base models first
    ensure_model("melspectrogram", OWW_BASE_MODEL_URLS, is_base=True)
    ensure_model("embedding_model", OWW_BASE_MODEL_URLS, is_base=True)

    # Setup wakeword model
    return ensure_model(model_name, OWW_MODEL_URLS)


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
    oww_model: Optional[OWWModel]
    oww_threshold: float
    min_speech_ms:int

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

        instance.min_speech_ms = int(attrs.get("min_speech_ms", DEFAULT_MIN_SPEECH_MS))


        # OpenWakeWord - if openwakeword_model is set, use it instead of Vosk
        oww_model_name = attrs.get("openwakeword_model", None)
        if oww_model_name:
            model_path = _setup_oww_models(oww_model_name, instance.logger)
            inference_framework = "tflite" if model_path.endswith(".tflite") else "onnx"
            instance.oww_model = OWWModel(
                wakeword_models=[model_path],
                inference_framework=inference_framework
            )
            instance.oww_threshold = float(attrs.get("openwakeword_threshold", 0.5))
            instance.logger.info(
                f"OpenWakeWord enabled with model={oww_model_name}, "
                f"threshold={instance.oww_threshold}"
            )
        else:
            instance.oww_model = None
            instance.oww_threshold = 0.5

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

        wake_words: Any = attrs.get("wake_words", [])
        if wake_words == []:
            raise ValueError("wake_words attribute is required")

        # Validate wake_words are strings
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
            wake_word_detected = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                self._check_for_wake_word,
                bytes(speech_buffer),
                AUDIO_SAMPLE_RATE_HZ,
            )

            if wake_word_detected:
                self.logger.info(
                    f"Wake word detected! Yielding {len(speech_chunk_buffer)} chunks ({len(speech_buffer)} bytes)"
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

    def _check_oww_wake_word(self, audio_bytes: bytes) -> bool:
        """
        Check if wake word is detected at start of audio using OpenWakeWord.

        Only checks the first 2 seconds to ensure wake word is at the beginning.
        """
        chunk_size = 2560  # 80ms at 16kHz (1280 samples * 2 bytes)
        # Only check first 2 seconds (32000 bytes at 16kHz, 16-bit)
        max_check_bytes = 32000
        check_bytes = min(len(audio_bytes), max_check_bytes)

        max_score = 0.0
        detected_word = None

        # Reset OWW model state for fresh detection
        self.oww_model.reset()

        # Process audio in chunks (only first 2 seconds)
        for i in range(0, check_bytes - chunk_size + 1, chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            prediction = self.oww_model.predict(audio_array)

            for wake_word, score in prediction.items():
                if score > max_score:
                    max_score = score
                    detected_word = wake_word
                if score >= self.oww_threshold:
                    self.logger.info(
                        f"OWW detected '{wake_word}' "
                        f"(score={score:.3f} >= {self.oww_threshold})"
                    )
                    return True

        # Check if wake word appears later - only warn if above threshold
        if len(audio_bytes) > max_check_bytes:
            for i in range(max_check_bytes, len(audio_bytes) - chunk_size + 1, chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                audio_array = np.frombuffer(chunk, dtype=np.int16)
                prediction = self.oww_model.predict(audio_array)
                time_sec = i / 32000
                for wake_word, score in prediction.items():
                    if score >= self.oww_threshold:
                        self.logger.warning(
                            f"OWW: '{wake_word}' at {time_sec:.1f}s "
                            f"(score={score:.3f}) - TRIGGERED LATE"
                        )
                        return True

        self.logger.debug(
            f"OWW max score in first 2s: {max_score:.3f} for '{detected_word}', "
            f"threshold: {self.oww_threshold}"
        )
        return False

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
            min_speech_frames =  self.min_speech_ms // 30

            async for audio_chunk in mic_stream:
                # Exit stream if shutting down
                if self.is_shutting_down:
                    self.logger.info("Stream ending due to shutdown")
                    break

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
                            f"Speech segment ended ({speech_frames} frames, "
                            f"{len(speech_buffer)} bytes), checking for wake word"
                        )
                        if self.oww_model:
                            # Run OWW on full segment
                            oww_detected = self._check_oww_wake_word(
                                bytes(speech_buffer)
                            )
                            if oww_detected:
                                self.logger.info(
                                    f"Yielding {len(speech_chunk_buffer)} chunks "
                                    f"({len(speech_buffer)} bytes) - OWW detected"
                                )
                                for chunk in speech_chunk_buffer:
                                    yield chunk
                                empty_response = AudioChunk()
                                empty_response.audio.audio_data = b""
                                yield empty_response
                            else:
                                self.logger.debug(
                                    f"OWW wake word not detected, discarding. "
                                    f"Threshold: {self.oww_threshold}"
                                )
                        else:
                            # Use Vosk for wake word detection
                            async for chunk in self._process_speech_segment(
                                speech_chunk_buffer, speech_buffer
                            ):
                                yield chunk
                    else:
                        self.logger.debug(
                            f"Ignoring false positive: only {speech_frames} frames detected"
                        )

                    # Clear buffers either way
                    speech_chunk_buffer.clear()
                    speech_buffer.clear()
                    audio_buffer.clear()
                    is_speech_active = False
                    silence_frames = 0
                    speech_frames = 0

                # Prevent buffer from growing too large, process when it gets to max size
                if len(speech_buffer) > MAX_BUFFER_SIZE_BYTES:
                    self.logger.debug("Buffer exceeded max size, processing")
                    if self.oww_model:
                        if self._check_oww_wake_word(bytes(speech_buffer)):
                            self.logger.info(
                                f"Yielding {len(speech_chunk_buffer)} chunks "
                                f"({len(speech_buffer)} bytes) - OWW (max buffer)"
                            )
                            for chunk in speech_chunk_buffer:
                                yield chunk
                            empty_response = AudioChunk()
                            empty_response.audio.audio_data = b""
                            yield empty_response
                    else:
                        async for chunk in self._process_speech_segment(
                            speech_chunk_buffer, speech_buffer
                        ):
                            yield chunk

                    speech_chunk_buffer.clear()
                    speech_buffer.clear()
                    audio_buffer.clear()
                    is_speech_active = False
                    silence_frames = 0
                    speech_frames = 0

            # Process any remaining buffered audio when stream ends
            if speech_chunk_buffer and speech_frames >= min_speech_frames:
                self.logger.debug(
                    f"Stream ended with {len(speech_buffer)} bytes buffered, processing"
                )
                if self.oww_model:
                    if self._check_oww_wake_word(bytes(speech_buffer)):
                        for chunk in speech_chunk_buffer:
                            yield chunk
                        empty_response = AudioChunk()
                        empty_response.audio.audio_data = b""
                        yield empty_response
                else:
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
            # Use grammar to constrain recognition to wake words for better accuracy
            grammar = json.dumps(self.wake_words)
            recognizer = KaldiRecognizer(self.vosk_model, sample_rate, grammar)
            recognizer.SetWords(True)  # Enable word-level confidence scores
            recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(recognizer.FinalResult())

            text = result.get("text", "").lower()

            # Check confidence if available (reject low-confidence matches)
            min_confidence = 0.7
            if "result" in result and result["result"]:
                avg_conf = sum(w.get("conf", 1.0) for w in result["result"]) / len(result["result"])
                self.logger.debug(f"Vosk confidence: {avg_conf:.2f}")
                if avg_conf < min_confidence:
                    self.logger.debug(f"Rejecting low confidence match: '{text}' (conf={avg_conf:.2f} < {min_confidence})")
                    return False

            if text:
                self.logger.debug(f"Recognized text: '{text}'")
            else:
                self.logger.debug("Vosk returned empty text, no speech recognized")
                return False

            for wake_word in self.wake_words:
                if self.fuzzy_matcher:
                    # Use fuzzy matching to match wake word
                    match_details = self.fuzzy_matcher.match(text, wake_word)
                    if match_details:
                        self.logger.info(
                            f"Wake word '{wake_word}' detected (fuzzy match: "
                            f"'{match_details['matched_text']}', distance={match_details['distance']})"
                        )
                        return True
                else:
                    # search for exact wake word match anywhere in text
                    pattern = rf"\b{re.escape(wake_word)}\b"
                    if re.search(pattern, text):
                        self.logger.info(f"Wake word '{wake_word}' detected in: '{text}'")
                        return True

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
        raise NotImplementedError()

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
