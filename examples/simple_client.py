"""
Simple voice assistant using Google STT and TTS.

No LLM needed - just basic question/answer responses.
"""

import asyncio
import os

from viam.robot.client import RobotClient
from viam.components.audio_in import AudioIn, AudioCodec
from viam.components.audio_out import AudioOut, AudioInfo
import speech_recognition as sr
from gtts import gTTS


class SimpleVoiceAssistant:
    """Super Simple voice assistant to tell you the date and time."""

    def __init__(self, robot: RobotClient, trigger_name: str = "filter", audioout_name: str = "speaker"):
        self.robot = robot
        self.trigger_name = trigger_name
        self.audioout_name = audioout_name
        self.recognizer = sr.Recognizer()
        self.tts_lang = 'en'
        self.trigger = None
        self.audioout = None


    async def start(self):
        """Initialize robot components."""
        self.trigger = AudioIn.from_robot(self.robot, self.trigger_name)
        self.audioout = AudioOut.from_robot(self.robot, self.audioout_name)
        print(f"Connected to filtered microphone: {self.trigger_name}")
        print(f"Connected to speaker: {self.audioout_name}")


    def speech_to_text(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Convert audio to text."""
        audio = sr.AudioData(audio_data, sample_rate, 2)
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except:
            return ""

    def get_response(self, user_text: str) -> str:
        """Generate response."""
        if not user_text:
            return "I didn't catch that."

        user_lower = user_text.lower()

        if "time" in user_lower:
            from datetime import datetime
            return f"It's {datetime.now().strftime('%I:%M %p')}"
        if "date" in user_lower or "today" in user_lower:
            from datetime import datetime
            return f"Today is {datetime.now().strftime('%A, %B %d')}"
        if "bye" in user_lower:
            return "Goodbye!"

        return f"I heard: {user_text}"

    async def speak(self, text: str):
        """Text to speech."""
        temp_mp3 = "/tmp/tts.mp3"
        gTTS(text=text, lang=self.tts_lang).save(temp_mp3)

        with open(temp_mp3, 'rb') as f:
            mp3_data = f.read()
        os.remove(temp_mp3)

        audio_info = AudioInfo(codec=AudioCodec.MP3, sample_rate_hz=24000, num_channels=1)
        await self.audioout.play(mp3_data, audio_info)

    async def run(self):
        """Continuously listen and respond."""
        print("="*60)
        print("VOICE ASSISTANT")
        print("="*60)
        print("Listening for trigger word...")
        print("(The trigger component will send audio when it detects the word)")
        print("="*60 + "\n")

        # Start continuous stream
        audio_stream = await self.trigger.get_audio("pcm16", 0, 0)

        audio_buffer = bytearray()
        chunk_count = 0

        try:
            async for response in audio_stream:
                audio_chunk = response.audio.audio_data

                if len(audio_chunk) == 0:
                    continue

                audio_buffer.extend(audio_chunk)
                chunk_count += 1

                # When trigger word detected, component sends burst of historical audio
                # Process when we have enough (e.g., 100KB ~ 3 seconds at 16kHz)
                if len(audio_buffer) >= 100000:
                    print(f"\nTrigger detected! Processing {len(audio_buffer)} bytes...")

                    user_text = self.speech_to_text(bytes(audio_buffer))

                    if user_text:
                        print(f"You: {user_text}")

                        # Get response
                        response_text = self.get_response(user_text)
                        print(f"Bot: {response_text}")

                        # Speak
                        await self.speak(response_text)
                    else:
                        print("No speech recognized")

                    # Clear buffer and continue listening
                    audio_buffer.clear()
                    chunk_count = 0
                    print("\nListening for next trigger...\n")

        except KeyboardInterrupt:
            print("\n\nStopping...")


async def main():
    opts = RobotClient.Options.with_api_key(
        api_key='<API_KEY>',
        api_key_id='<API_KEY_ID>'
    )
    robot =  await RobotClient.at_address(<ADDRESS>, opts)

    try:
        assistant = SimpleVoiceAssistant(robot, "filter", "speaker")
        await assistant.start()
        await assistant.run()
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
