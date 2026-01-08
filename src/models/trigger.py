
from typing import ClassVar, Mapping, Optional, Sequence, Tuple, cast
import asyncio
import json
import os
import urllib.request
import zipfile
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


def download_vosk_model(model_name: str = "vosk-model-small-en-us-0.15", logger=None) -> str:
    """
    Download Vosk model if not present.

    Args:
        model_name: Name of the Vosk model to download
        logger: Optional logger instance

    Returns:
        Path to the model directory
    """
    model_dir = os.path.expanduser(f"~/{model_name}")

    if os.path.exists(model_dir):
        if logger:
            logger.info(f"Vosk model already exists at {model_dir}")
        return model_dir

    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    zip_path = f"/tmp/{model_name}.zip"

    if logger:
        logger.info(f"Downloading Vosk model from {url}...")
    else:
        print(f"Downloading Vosk model from {url}...")

    try:
        urllib.request.urlretrieve(url, zip_path)

        if logger:
            logger.info(f"Extracting to {model_dir}...")
        else:
            print(f"Extracting to {model_dir}...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.expanduser("~"))

        os.remove(zip_path)

        if logger:
            logger.info(f"Vosk model downloaded successfully to {model_dir}")
        else:
            print(f"Vosk model downloaded successfully to {model_dir}")

        return model_dir

    except Exception as e:
        if logger:
            logger.error(f"Failed to download Vosk model: {e}")
        raise RuntimeError(f"Failed to download Vosk model: {e}")


class Trigger(AudioIn, EasyResource):
    """Simplified trigger component without hearken."""

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "filter-mic"), "trigger")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        """Create new trigger component."""
        instance = super().new(config, dependencies)
        instance.logger.info("=== Simple Trigger Init ===")

        attrs = struct_to_dict(config.attributes)
        microphone = str(attrs.get("source_microphone", ""))
        instance.trigger_word = str(attrs.get("trigger_word", "")).lower()
        model_path = str(attrs.get("vosk_model_path", "~/vosk-model-small-en-us-0.15"))
        vad_aggressiveness = int(attrs.get("vad_aggressiveness", 3))  # 0-3, higher = less sensitive

        instance.logger.info(f"Trigger word: '{instance.trigger_word}'")
        instance.logger.info(f"VAD aggressiveness: {vad_aggressiveness}")

        # Initialize WebRTC VAD (lightweight!)
        instance.vad = webrtcvad.Vad(vad_aggressiveness)
        instance.logger.info("WebRTC VAD initialized")

        # Load Vosk model (download if needed)
        model_path = os.path.expanduser(model_path)

        # If path doesn't exist, try to download the default model
        if not os.path.exists(model_path):
            instance.logger.info(f"Vosk model not found at {model_path}, attempting download...")
            # Extract model name from path
            model_name = os.path.basename(model_path)
            try:
                model_path = download_vosk_model(model_name, instance.logger)
            except Exception as e:
                instance.logger.error(f"Failed to download model: {e}")
                raise RuntimeError(f"Vosk model not found at {model_path} and download failed: {e}")

        instance.vosk_model = VoskModel(model_path)
        instance.logger.info("Vosk model loaded")

        # Create thread pool for Vosk processing (non-blocking)
        instance.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vosk")
        instance.logger.info("Thread pool executor created for Vosk processing")

        # Get microphone
        if microphone:
            mic = dependencies[AudioIn.get_resource_name(microphone)]
            instance.microphone_client = cast(AudioIn, mic)
            instance.logger.info(f"Microphone: {microphone}")
        else:
            instance.microphone_client = None

        instance.logger.info("=== Init Complete ===")
        return instance

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        """Validate configuration."""
        deps = []
        attrs = struct_to_dict(config.attributes)
        microphone = str(attrs.get("source_microphone", ""))
        if microphone:
            deps.append(microphone)
        return deps, []

    def check_for_trigger(self, audio_bytes: bytes, sample_rate: int = 16000) -> bool:
        """
        Check if trigger word is in audio.

        Args:
            audio_bytes: Raw PCM16 audio data
            sample_rate: Audio sample rate

        Returns:
            bool: True if trigger word detected
        """
        try:
            self.logger.debug(f"check_for_trigger: {len(audio_bytes)} bytes, {sample_rate} Hz")

            recognizer = KaldiRecognizer(self.vosk_model, sample_rate)

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
                self.logger.warning("Vosk returned EMPTY text - no speech recognized!")

            if self.trigger_word and self.trigger_word in text:
                self.logger.info(f"TRIGGER WORD '{self.trigger_word}' DETECTED!")
                return True

            return False
        except Exception as e:
            self.logger.error(f"Vosk error: {e}", exc_info=True)
            return False

    async def get_audio(self, codec: str, duration_seconds: float, previous_timestamp_ns: int, **kwargs):
        """
        Stream audio, yielding buffered chunks when trigger word detected.

        Uses WebRTC VAD to detect speech (low CPU), then Vosk to find trigger.

        Args:
            codec: Audio codec (should be "pcm16")
            duration_seconds: Duration (use 0 for continuous)
            previous_timestamp_ns: Previous timestamp (use 0 to start)

        Yields:
            AudioResponse chunks with original timestamps
        """
        if not self.microphone_client:
            self.logger.error("No microphone configured")
            return

        async def audio_generator():
            self.logger.info("Starting trigger detection with VAD...")

            # Check mic properties
            #mic_props = await self.microphone_client.get_properties()
            #self.logger.info(f"Microphone properties - Sample rate: {mic_props.sample_rate_hz} Hz, Channels: {mic_props.channel_count}, Codec: {codec}")

            # Get continuous stream from microphone
            mic_stream = await self.microphone_client.get_audio(codec, 0, 0)

            # Buffers
            chunk_buffer = []  # AudioResponse objects (with timestamps!)
            byte_buffer = bytearray()

            # Speech detection state
            is_speech_active = False
            silence_frames = 0
            max_silence_frames = 30  # ~1 second of silence to end speech

            async for response in mic_stream:
                audio_data = response.audio.audio_data

                if not audio_data:
                    continue

                # Check PCM16 alignment (2 bytes per sample)
                if len(audio_data) % 2 != 0:
                    self.logger.warning(f"Misaligned audio chunk detected: {len(audio_data)} bytes (odd length)")

                # WebRTC VAD requires specific frame sizes (10, 20, or 30ms)
                # At 16kHz: 30ms = 480 samples = 960 bytes
                frame_size = 960

                # Track if we should process this batch
                should_process = False

                # Process audio in VAD-compatible frames
                for i in range(0, len(audio_data), frame_size):
                    frame = audio_data[i:i + frame_size]

                    if len(frame) < frame_size:
                        continue  # Skip incomplete frames

                    # Check if frame contains speech (vad is cheap CPU)
                    try:
                        is_speech = self.vad.is_speech(frame, 16000)
                    except:
                        is_speech = False

                    if is_speech:
                        if not is_speech_active:
                            self.logger.debug("Speech started")
                            is_speech_active = True
                        silence_frames = 0
                    else:
                        if is_speech_active:
                            silence_frames += 1

                            if silence_frames >= max_silence_frames:
                                self.logger.debug(f"Speech ended ({silence_frames} silent frames)")
                                should_process = True
                                break  # Exit frame loop

                # Only buffer during active speech (not during silence)
                if is_speech_active:
                    chunk_buffer.append(response)
                    byte_buffer.extend(audio_data)

                # If speech ended, check for trigger
                if should_process:
                    self.logger.debug(f"Checking {len(byte_buffer)} bytes for trigger")

                    # Run Vosk in thread pool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    audio_to_check = bytes(byte_buffer)
                    trigger_detected = await loop.run_in_executor(
                        self.executor,
                        self.check_for_trigger,
                        audio_to_check
                    )

                    if trigger_detected:
                        # Trigger detected! Yield all buffered chunks
                        self.logger.info(f"TRIGGER! Yielding {len(chunk_buffer)} chunks ({len(byte_buffer)} bytes)")

                        for chunk in chunk_buffer:
                            yield chunk

                        self.logger.info("Ready for next trigger")
                    else:
                        self.logger.debug("No trigger found")

                    # Clear buffers either way
                    chunk_buffer.clear()
                    byte_buffer.clear()
                    is_speech_active = False
                    silence_frames = 0

                # Prevent buffers from growing too large (safety)
                if len(byte_buffer) > 500000:  # ~15 seconds max
                    self.logger.warning("Buffer too large, force checking")

                    # Run Vosk in thread pool (non-blocking)
                    loop = asyncio.get_event_loop()
                    audio_to_check = bytes(byte_buffer)
                    trigger_detected = await loop.run_in_executor(
                        self.executor,
                        self.check_for_trigger,
                        audio_to_check
                    )

                    if trigger_detected:
                        self.logger.info(f"TRIGGER! Yielding {len(chunk_buffer)} chunks")
                        for chunk in chunk_buffer:
                            yield chunk

                    chunk_buffer.clear()
                    byte_buffer.clear()
                    is_speech_active = False
                    silence_frames = 0

        return StreamWithIterator(audio_generator())

    async def close(self):
        """Clean up resources."""
        self.logger.info("Closing trigger component")

    async def do_command(self, command, **kwargs):
        raise NotImplementedError()

    async def get_geometries(self, **kwargs):
        raise NotImplementedError()

    async def get_properties(self, **kwargs):
        self.logger.debug("get_properties called")
        # Return properties from underlying microphone
        if self.microphone_client:
            return await self.microphone_client.get_properties()
        return AudioIn.Properties(sample_rate_hz=16000, channel_count=1)
