"""
Voice assistant using Google Speech-to-Text, OpenAI, and ElevenLabs TTS.

Requires: pip install viam-sdk openai SpeechRecognition elevenlabs

Set environment variables:
    OPENAI_API_KEY - Get from https://platform.openai.com/api-keys
    ELEVENLABS_API_KEY - Get from https://elevenlabs.io/app/settings/api-keys
"""

import asyncio
import os
import re

from viam.robot.client import RobotClient
from viam.components.audio_in import AudioIn, AudioCodec
from viam.components.audio_out import AudioOut, AudioInfo
import speech_recognition as sr
from openai import OpenAI
from elevenlabs import ElevenLabs


class ElevenLabsVoiceAssistant:
    """Voice assistant powered by Google STT, OpenAI, and ElevenLabs TTS."""

    def __init__(
        self,
        robot: RobotClient,
        filter_name: str = "filter",
        audioout_name: str = "speaker",
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",  # Default: George
        completion_model: str = "gpt-4o",
        wake_words: list[str] = None,
        persona: str = None,
    ):
        self.robot = robot
        self.filter_name = filter_name
        self.audioout_name = audioout_name
        self.voice_id = voice_id
        self.completion_model = completion_model
        self.wake_words = [w.lower() for w in (wake_words or ["hey robot"])]
        self.persona = persona
        self.recognizer = sr.Recognizer()
        self.filter = None
        self.audioout = None

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_history = []

        # Build system prompt
        if persona:
            self.system_prompt = f"You are {persona}. Respond in character. Keep responses concise and conversational."
        else:
            self.system_prompt = "You are a helpful voice assistant. Keep responses concise and conversational."

        # Initialize ElevenLabs client
        self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    async def start(self):
        self.filter = AudioIn.from_robot(self.robot, self.filter_name)
        self.audioout = AudioOut.from_robot(self.robot, self.audioout_name)
        print(f"Connected to filtered microphone: {self.filter_name}")
        print(f"Connected to speaker: {self.audioout_name}")

    def speech_to_text(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Convert audio to text using Google Speech Recognition."""
        audio = sr.AudioData(audio_data, sample_rate, 2)
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except:
            return ""

    def strip_wake_word(self, text: str) -> str:
        """Remove wake word from the beginning of transcribed text."""
        text_lower = text.lower()
        for wake_word in self.wake_words:
            # Match wake word at start with optional punctuation/whitespace after
            pattern = rf"^{re.escape(wake_word)}[,.\s]*"
            match = re.match(pattern, text_lower)
            if match:
                return text[match.end():].strip()
        return text

    def get_response(self, user_text: str) -> str:
        """Generate response using OpenAI."""
        if not user_text:
            return "I didn't catch that."

        try:
            # Build messages for chat completion
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.chat_history)
            messages.append({"role": "user", "content": user_text})

            # Send message to OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.completion_model,
                messages=messages,
            )

            assistant_message = response.choices[0].message.content

            # Update chat history
            self.chat_history.append({"role": "user", "content": user_text})
            self.chat_history.append({"role": "assistant", "content": assistant_message})

            return assistant_message
        except Exception as e:
            print(f"Error getting OpenAI response: {e}")
            return "Sorry, I had trouble processing that."

    async def speak(self, text: str):
        """Text to speech using ElevenLabs."""
        try:
            # Generate audio using ElevenLabs
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id="eleven_turbo_v2_5",
                output_format="pcm_16000",
            )

            # Collect audio bytes from generator
            pcm_data = b"".join(audio_generator)

            audio_info = AudioInfo(codec=AudioCodec.PCM16, sample_rate_hz=16000, num_channels=1)
            await self.audioout.play(pcm_data, audio_info)
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")

    async def run(self):
        """Continuously listen and respond."""
        print("Listening for wake word...")

        while True:
            # Start continuous stream
            try:
                audio_stream = await self.filter.get_audio("pcm16", 0, 0)
            except Exception as e:
                print(f"Error starting audio stream: {e}, retrying...")
                await asyncio.sleep(1)
                continue

            try:
                segment = bytearray()

                async for chunk in audio_stream:
                    audio_data = chunk.audio.audio_data

                    if len(audio_data) == 0:
                        # Empty chunk = segment ended, process it
                        if segment:
                            print(f"\nWake word detected! Processing {len(segment)} bytes...")

                            full_text = self.speech_to_text(bytes(segment))

                            if full_text:
                                # Strip wake word from transcription
                                user_text = self.strip_wake_word(full_text)
                                print(f"Heard: {full_text}")
                                print(f"Query: {user_text}")
                                response_text = self.get_response(user_text)
                                print(f"Response: {response_text}")
                                await self.speak(response_text)
                            else:
                                print("No speech recognized")

                            segment.clear()
                            print("Listening for next wake word...\n")
                    else:
                        # Accumulate audio data
                        segment.extend(audio_data)

            except KeyboardInterrupt:
                print("\n\nStopping...")
                return
            except Exception as e:
                print(f"Stream disconnected: {e}, reconnecting...")
                await asyncio.sleep(1)
                continue


async def main():
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        print("Get a key at: https://platform.openai.com/api-keys")
        return

    if not os.getenv("ELEVENLABS_API_KEY"):
        print("Error: ELEVENLABS_API_KEY environment variable not set")
        print("Set it with: export ELEVENLABS_API_KEY='your-api-key'")
        print("Get a key at: https://elevenlabs.io/app/settings/api-keys")
        return

    # Connect to your robot - update these credentials
    opts = RobotClient.Options.with_api_key(
        api_key='yn2v121yklqjfduvse718fn582dskzi1',
        api_key_id='d04e49b3-7799-4afe-ba3a-a5b35d802b17'
    )

    robot = await RobotClient.at_address('xarm-main.aqb785vhl4.viam.cloud', opts)

    try:
        assistant = ElevenLabsVoiceAssistant(
            robot,
            filter_name="filter",
            audioout_name="speaker",
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            completion_model="gpt-4o",
            wake_words=["hey robot"],  # Must match filter config
            persona=None,
        )
        await assistant.start()
        await assistant.run()
    finally:
        await robot.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user")
