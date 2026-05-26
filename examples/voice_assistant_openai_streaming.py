"""
Streaming voice assistant using OpenAI Whisper, GPT, and OpenAI TTS.

Same flow as voice_assistant_openai.py but uses audio_out.play_stream for the
TTS response: PCM chunks are pushed to the speaker as soon as they arrive
from OpenAI's TTS streaming endpoint, so playback starts within a few hundred
milliseconds instead of waiting for the entire response to download.

Requires: pip install viam-sdk openai
Set environment variable: OPENAI_API_KEY
"""

import asyncio
import os
import wave
from io import BytesIO
from typing import AsyncIterator

from viam.robot.client import RobotClient
from viam.components.audio_in import AudioIn, AudioCodec
from viam.components.audio_out import AudioOut, AudioInfo
from openai import OpenAI


class OpenAIStreamingVoiceAssistant:
    """Voice assistant that pipes streaming TTS straight into play_stream."""

    # OpenAI TTS streaming returns 24 kHz mono PCM16.
    TTS_SAMPLE_RATE = 24000
    TTS_CHANNELS = 1

    def __init__(
        self,
        robot: RobotClient,
        filter_name: str = "filter",
        audioout_name: str = "speaker",
    ):
        self.robot = robot
        self.filter_name = filter_name
        self.audioout_name = audioout_name
        self.filter = None
        self.audioout = None

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = (
            "You are a helpful voice assistant. "
            "Keep responses concise and conversational."
        )
        self.chat_history = []

    async def start(self):
        self.filter = AudioIn.from_robot(self.robot, self.filter_name)
        self.audioout = AudioOut.from_robot(self.robot, self.audioout_name)
        print(f"Connected to filtered microphone: {self.filter_name}")
        print(f"Connected to speaker: {self.audioout_name}")

    def speech_to_text(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"
        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer,
        )
        return response.text

    def get_response(self, user_text: str) -> str:
        if not user_text:
            return "I didn't catch that."

        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.chat_history)
            messages.append({"role": "user", "content": user_text})

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )

            assistant_message = response.choices[0].message.content
            self.chat_history.append({"role": "user", "content": user_text})
            self.chat_history.append(
                {"role": "assistant", "content": assistant_message}
            )
            return assistant_message
        except Exception as e:
            print(f"Error getting GPT response: {e}")
            return "Sorry, I had trouble processing that."

    async def _tts_pcm_chunks(self, text: str) -> AsyncIterator[bytes]:
        """
        Bridge OpenAI's synchronous streaming TTS response into an async
        generator of raw PCM16 byte chunks. Each yielded value is a chunk of
        audio bytes ready to be sent to play_stream.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=8)

        def producer():
            try:
                with self.client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="alloy",
                    input=text,
                    response_format="pcm",  # 24 kHz mono PCM16
                ) as response:
                    for chunk in response.iter_bytes(chunk_size=4096):
                        if chunk:
                            asyncio.run_coroutine_threadsafe(
                                queue.put(chunk), loop
                            ).result()
            except Exception as e:
                print(f"TTS producer error: {e}")
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        # Run the blocking OpenAI iterator on a thread so the event loop stays free.
        producer_task = loop.run_in_executor(None, producer)

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            await producer_task

    async def speak(self, text: str):
        """
        Stream TTS output to the speaker as chunks arrive. play_stream blocks
        until the chunks generator is exhausted and the speaker has drained.
        """
        info = AudioInfo(
            codec=AudioCodec.PCM_16,
            sample_rate_hz=self.TTS_SAMPLE_RATE,
            num_channels=self.TTS_CHANNELS,
        )
        try:
            await self.audioout.play_stream(info, self._tts_pcm_chunks(text))
        except Exception as e:
            print(f"Error in streaming TTS: {e}")

    async def run(self):
        print("Listening for wake word...")

        while True:
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
                        if segment:
                            print(
                                f"\nWake word detected! "
                                f"Processing {len(segment)} bytes..."
                            )
                            try:
                                user_text = self.speech_to_text(bytes(segment))
                                if user_text:
                                    print(f"You: {user_text}")
                                    response_text = self.get_response(user_text)
                                    print(f"Bot: {response_text}")
                                    await self.speak(response_text)
                                else:
                                    print("No speech recognized")
                            except Exception as e:
                                print(f"Error processing speech: {e}")

                            segment.clear()
                            print("Listening for next wake word...\n")
                    else:
                        segment.extend(audio_data)

            except KeyboardInterrupt:
                print("\n\nStopping...")
                return
            except Exception as e:
                print(f"Stream disconnected: {e}, reconnecting...")
                await asyncio.sleep(1)
                continue


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return

    opts = RobotClient.Options.with_api_key(api_key='', api_key_id='')
    robot = await RobotClient.at_address('', opts)

    try:
        assistant = OpenAIStreamingVoiceAssistant(robot, "filter", "speaker")
        await assistant.start()
        await assistant.run()
    finally:
        await robot.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped by user")
