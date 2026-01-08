
from typing import ClassVar, Mapping, Optional, Sequence, Tuple, cast
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from vosk import Model as VoskModel, KaldiRecognizer
import webrtcvad
from typing_extensions import Self
from viam.components.audio_in import AudioIn
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict
from viam.streams import StreamWithIterator

from .vosk import download_vosk_model

# Default configuration values
DEFAULT_VOSK_MODEL = "vosk-model-small-en-us-0.15"
DEFAULT_VAD_AGGRESSIVENESS = 3  # 0-3, higher = less sensitive

class WakeWordFilter(AudioIn, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "filtered-audio"), "wake-word-filter")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
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
        vad_aggressiveness = int(attrs.get("vad_aggressiveness", DEFAULT_VAD_AGGRESSIVENESS))

        # Initialize WebRTC VAD
        instance.vad = webrtcvad.Vad(vad_aggressiveness)
        instance.logger.debug("WebRTC VAD initialized")

        # Load Vosk model (download if needed)
        data_path = os.getenv("VIAM_MODULE_DATA")
        model_path = data_path + "/" + model

        # If path doesn't exist, try to download the model
        if not os.path.exists(model_path):
            instance.logger.info(f"Vosk model not found at {model_path}, attempting download...")
            try:
                model_path = download_vosk_model(model, instance.logger)
            except Exception as e:
                instance.logger.error(f"Failed to download model: {e}")
                raise RuntimeError(f"Vosk model not found at {model_path} and download failed: {e}")

        instance.vosk_model = VoskModel(model_path)
        instance.logger.debug("Vosk model loaded")

        # Create thread pool for Vosk processing (non-blocking)
        instance.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vosk")
        instance.logger.debug("Thread pool executor created for Vosk processing")

        # Get source microphone
        if microphone:
            mic = dependencies[AudioIn.get_resource_name(microphone)]
            instance.microphone_client = cast(AudioIn, mic)
        else:
            instance.logger.error("wake-word-filter must have a source microphone, no microhone found")
            raise RuntimeError("wake-word-filter must have a source microphone, no microhone found")

        return instance

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        deps = []
        attrs = struct_to_dict(config.attributes)
        mic = attrs.get("source_microphone", "")

        if mic == "":
            raise RuntimeError("source_microphone attribute is required")
        if not isinstance(mic, str):
            raise RuntimeError("source_microphone attribute must be a string")
        deps.append(mic)

        wake_words = attrs.get("wake_words", [])
        if wake_words == []:
            raise RuntimeError("wake_words attribute is required")

        # Validate wake_words are strings
        if isinstance(wake_words, str):
            # Single string is valid
            pass
        elif isinstance(wake_words, list):
            # List must contain only strings
            for word in wake_words:
                if not isinstance(word, str):
                    raise RuntimeError(f"All wake_words must be strings, got {type(word).__name__}")
        else:
            raise RuntimeError(f"wake_words must be a string or list of strings, got {type(wake_words).__name__}")

        return deps, []

    async def get_audio(self, codec: str, duration_seconds: float, previous_timestamp_ns: int, **kwargs):
        """
        Stream audio, yielding buffered audio chunks when a wake word detected.

        Uses WebRTC VAD to detect speech, then Vosk to find wake words.

        Args:
            codec: Audio codec (should be "pcm16")
            duration_seconds: Duration (use 0 for continuous)
            previous_timestamp_ns: Previous timestamp (use 0 to start from now)

        Yields:
            AudioResponse chunks with original timestamps
        """
        if not self.microphone_client:
            raise ValueError("no source microphone found")

        if codec.lower() != "pcm16":
          self.logger.error(f"Unsupported codec: {codec}. Only PCM16 is supported.")
          raise ValueError(f"Wake word filter only supports PCM16 codec, got: {codec}")

        async def audio_generator():
            self.logger.info("Starting speech detection with VAD...")

            # Check mic properties
            mic_props = await self.microphone_client.get_properties()
            self.logger.info(f"Microphone properties - Sample rate: {mic_props.sample_rate_hz} Hz, Channels: {mic_props.num_channels}")

            # Validate sample rate (Vosk models are trained on 16000 Hz)
            if mic_props.sample_rate_hz != 16000:
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

            # Get audio stream from source microphone
            mic_stream = await self.microphone_client.get_audio(codec, duration_seconds, previous_timestamp_ns)

            chunk_buffer = []
            byte_buffer = bytearray()

            is_speech_active = False
            silence_frames = 0
            max_silence_frames = 30  # ~1 second of silence to end speech segment

            async for audio_chunk in mic_stream:
                audio_data = audio_chunk.audio.audio_data

                if not audio_data:
                    continue

                # WebRTC VAD requires specific frame sizes (10, 20, or 30ms)
                # At 16kHz: 30ms = 480 samples = 960 bytes
                frame_size = 960

                # Track if we should process this batch
                should_process = False

                # Process audio bytes in 30ms chunks
                for i in range(0, len(audio_data), frame_size):
                    frame = audio_data[i:i + frame_size]

                    if len(frame) < frame_size:
                        continue  # Skip incomplete frames

                    # Check if frame contains speech
                    try:
                        # The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, or 32000 Hz.
                        #  A frame must be either 10, 20, or 30 ms in duration
                        is_speech = self.vad.is_speech(frame, mic_props.sample_rate_hz)
                    except Exception as e:
                        self.logger.error(f"VAD error: {e}")
                        is_speech = False

                    if is_speech:
                        if not is_speech_active:
                            self.logger.debug("Speech segment started")
                            is_speech_active = True
                        silence_frames = 0
                    else:
                        if is_speech_active:
                            silence_frames += 1

                            if silence_frames >= max_silence_frames:
                                self.logger.debug(f"Speech segment ended ({silence_frames} silent frames)")
                                should_process = True
                                break

                # Only buffer during active speech
                if is_speech_active:
                    chunk_buffer.append(audio_chunk)
                    byte_buffer.extend(audio_data)

                # If speech segment ended, check for wake word
                if should_process:
                    self.logger.debug(f"Checking {len(byte_buffer)} bytes for wake word")

                    # Run Vosk in thread pool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    audio_to_check = bytes(byte_buffer)
                    wake_word_detected = await loop.run_in_executor(
                        self.executor,
                        self.check_for_wake_word,
                        audio_to_check,
                        16000,
                    )

                    if wake_word_detected:
                        self.logger.info(f"Wake word detected! Yielding {len(chunk_buffer)} chunks ({len(byte_buffer)} bytes)")

                        for chunk in chunk_buffer:
                            yield chunk
                    else:
                        self.logger.debug("No wake words found")

                    # Clear buffers either way
                    chunk_buffer.clear()
                    byte_buffer.clear()
                    is_speech_active = False
                    silence_frames = 0

                # Prevent buffer from growing too large
                if len(byte_buffer) > 500000:  # ~15 seconds max
                    self.logger.warning("Buffer too large, force checking")

                    # Run Vosk in thread pool (non-blocking)
                    loop = asyncio.get_event_loop()
                    audio_to_check = bytes(byte_buffer)
                    trigger_detected = await loop.run_in_executor(
                        self.executor,
                        self.check_for_wake_word,
                        audio_to_check,
                        16000
                    )

                    if trigger_detected:
                        self.logger.info(f"wake word detected! Yielding {len(chunk_buffer)} chunks")
                        for chunk in chunk_buffer:
                            yield chunk

                    chunk_buffer.clear()
                    byte_buffer.clear()
                    is_speech_active = False
                    silence_frames = 0

        return StreamWithIterator(audio_generator())


    def check_for_wake_word(self, audio_bytes: bytes, sample_rate) -> bool:
        """
        Check if any wake word is in audio.

        Args:
            audio_bytes: Raw PCM16 audio data
            sample_rate: Audio sample rate

        Returns:
            bool: True if any wake word detected
        """
        try:
            self.logger.debug(f"check_for_wake_word got: {len(audio_bytes)} bytes at {sample_rate} Hz")

            grammar = json.dumps(self.wake_words)
            recognizer = KaldiRecognizer(self.vosk_model, sample_rate, grammar)


            # Process audio
            accepted = recognizer.AcceptWaveform(audio_bytes)
            self.logger.debug(f"AcceptWaveform returned: {accepted}")

            # Get final result
            result = json.loads(recognizer.FinalResult())
            self.logger.info(f"Vosk full result: {result}")

            text = result.get("text", "").lower()

            if text:
                self.logger.info(f"Recognized text: '{text}'")
            else:
                self.logger.debug("Vosk returned empty text, no speech recongized")

            # Check if any wake word is in the recognized text
            for wake_word in self.wake_words:
                if wake_word in text:
                    self.logger.debug(f"Wake word '{wake_word}' detected")
                    return True

            return False
        except Exception as e:
            self.logger.error(f"Vosk error: {e}", exc_info=True)
            return False

    async def close(self):
        raise NotImplementedError()

    async def do_command(self, command, **kwargs):
        raise NotImplementedError()

    async def get_geometries(self, **kwargs):
        raise NotImplementedError()

    async def get_properties(self, **kwargs):
        # Return properties from underlying microphone
        props =  await self.microphone_client.get_properties()
        # Ensure we only report PCM16 support
        return AudioIn.Properties(
            supported_codecs = ["pcm16"],
            sample_rate_hz=props.sample_rate_hz,
            num_channels=props.num_channels
        )
