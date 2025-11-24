import os
import asyncio
import logging
import threading
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV1SocketClientResponse,
    SpeakV1SocketClientResponse,
    SpeakV1TextMessage,
    SpeakV1ControlMessage,
)
from deepgram.extensions.types.sockets.listen_v1_results_event import (
    ListenV1ResultsEvent,
)
from openai import OpenAI
from dotenv import load_dotenv
import pyaudio

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16


class VoiceAssistant:
    def __init__(self):
        self.deepgram_stt = None
        self.deepgram_connection = None
        self.is_processing = False
        self.current_transcript = ""
        self.event_loop = None
        self.audio = pyaudio.PyAudio()
        self.is_listening = False

    async def connect_deepgram_stt(self):
        try:
            self.event_loop = asyncio.get_event_loop()

            self.deepgram_connection = deepgram.listen.v1.connect(
                model="nova-2",
                language="en-US",
                smart_format="true",
                interim_results="true",
                encoding="linear16",
                sample_rate=str(SAMPLE_RATE),
                channels=str(CHANNELS),
            )

            self.deepgram_stt = self.deepgram_connection.__enter__()

            assistant = self

            def on_message(message: ListenV1SocketClientResponse) -> None:
                try:
                    if isinstance(message, ListenV1ResultsEvent):
                        if hasattr(message, "channel") and message.channel:
                            if (
                                hasattr(message.channel, "alternatives")
                                and message.channel.alternatives
                            ):
                                sentence = message.channel.alternatives[0].transcript
                                if len(sentence.strip()) > 0:
                                    is_final = getattr(message, "is_final", True)
                                    if is_final:
                                        assistant.current_transcript = sentence
                                        logger.info(f"Final transcript: {sentence}")
                                        if (
                                            assistant.event_loop
                                            and assistant.event_loop.is_running()
                                        ):
                                            asyncio.run_coroutine_threadsafe(
                                                assistant.process_with_openai(),
                                                assistant.event_loop,
                                            )
                                        else:
                                            logger.warning(
                                                "Event loop not available, cannot process with OpenAI"
                                            )
                                    else:
                                        logger.debug(f"Interim: {sentence}")
                except Exception as e:
                    logger.error(f"Error in on_message handler: {e}")

            def on_error(error):
                logger.error(f"Deepgram STT error: {error}")

            def on_close(_):
                logger.info("Deepgram STT connection closed")
                assistant.deepgram_stt = None

            self.deepgram_stt.on(
                EventType.OPEN, lambda _: logger.info("Deepgram STT connection opened")
            )
            self.deepgram_stt.on(EventType.MESSAGE, on_message)
            self.deepgram_stt.on(EventType.ERROR, on_error)
            self.deepgram_stt.on(EventType.CLOSE, on_close)

            def listening_thread():
                try:
                    if self.deepgram_stt:
                        self.deepgram_stt.start_listening()
                except Exception as e:
                    logger.error(f"Error in listening thread: {e}")
                    assistant.deepgram_stt = None

            listen_thread = threading.Thread(target=listening_thread, daemon=True)
            listen_thread.start()

            await asyncio.sleep(0.5)

            logger.info("Deepgram STT connected")
            return True

        except Exception as e:
            logger.error(f"Error connecting to Deepgram STT: {e}")
            self.deepgram_stt = None
            self.deepgram_connection = None
            return False

    def start_microphone_capture(self):
        self.is_listening = True

        def capture_thread():
            try:
                stream = self.audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                )

                logger.info("Microphone started. Speak now...")

                while self.is_listening:
                    try:
                        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        if self.deepgram_stt:
                            self.deepgram_stt.send_media(data)
                    except Exception as e:
                        if "closed" not in str(e).lower():
                            logger.error(f"Error sending audio: {e}")
                        break

                stream.stop_stream()
                stream.close()
                logger.info("Microphone stopped")

            except Exception as e:
                logger.error(f"Error in microphone capture: {e}")

        capture_thread_obj = threading.Thread(target=capture_thread, daemon=True)
        capture_thread_obj.start()
        return capture_thread_obj

    async def process_with_openai(self):
        if not self.current_transcript.strip() or self.is_processing:
            return

        self.is_processing = True
        transcript = self.current_transcript.strip()
        self.current_transcript = ""

        try:
            logger.info(f"Processing with OpenAI: {transcript}")

            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Keep responses concise and conversational, suitable for voice interaction.",
                    },
                    {"role": "user", "content": transcript},
                ],
                stream=True,
                temperature=0.7,
                max_tokens=150,
            )

            response_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response_text += content

            if response_text:
                logger.info(f"OpenAI response: {response_text}")
                await self.stream_to_tts(response_text)

        except Exception as e:
            logger.error(f"Error processing with OpenAI: {e}")
        finally:
            self.is_processing = False

    async def stream_to_tts(self, text: str):
        try:
            connection = deepgram.speak.v1.connect(
                model="aura-asteria-en",
                encoding="linear16",
                sample_rate=str(SAMPLE_RATE),
            )

            with connection as conn:
                audio_chunks = []
                audio_received = threading.Event()

                def on_message(message: SpeakV1SocketClientResponse) -> None:
                    if isinstance(message, bytes):
                        audio_chunks.append(message)
                        audio_received.set()
                    else:
                        msg_type = getattr(message, "type", "Unknown")
                        logger.debug(f"Received {msg_type} event from TTS")

                conn.on(EventType.OPEN, lambda _: logger.debug("TTS connection opened"))
                conn.on(EventType.MESSAGE, on_message)
                conn.on(
                    EventType.CLOSE, lambda _: logger.debug("TTS connection closed")
                )
                conn.on(
                    EventType.ERROR, lambda error: logger.error(f"TTS error: {error}")
                )

                def listening_thread():
                    try:
                        conn.start_listening()
                    except Exception as e:
                        logger.error(f"Error in TTS listening thread: {e}")

                listen_thread = threading.Thread(target=listening_thread, daemon=True)
                listen_thread.start()

                await asyncio.sleep(0.1)

                conn.send_text(SpeakV1TextMessage(type="Speak", text=text))

                conn.send_control(SpeakV1ControlMessage(type="Flush"))

                audio_received.wait(timeout=5.0)

                await asyncio.sleep(1.0)

                conn.send_control(SpeakV1ControlMessage(type="Close"))

                await asyncio.sleep(0.5)

                if audio_chunks:
                    audio_bytes = b"".join(audio_chunks)
                    self.play_audio(audio_bytes)

        except Exception as e:
            logger.error(f"Error in TTS streaming: {e}")

    def play_audio(self, audio_data: bytes):
        try:
            stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE,
            )

            chunk_size = CHUNK_SIZE * 2
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if chunk:
                    stream.write(chunk)

            stream.stop_stream()
            stream.close()
            logger.info("Audio playback completed")

        except Exception as e:
            logger.error(f"Error playing audio: {e}")

    def stop(self):
        self.is_listening = False

        if self.deepgram_connection:
            try:
                self.deepgram_connection.__exit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing Deepgram connection: {e}")
            finally:
                self.deepgram_stt = None
                self.deepgram_connection = None

        self.audio.terminate()


async def main():
    required_vars = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        logger.error("Please set them in a .env file or environment")
        return

    assistant = VoiceAssistant()

    try:
        if not await assistant.connect_deepgram_stt():
            logger.error("Failed to connect to Deepgram STT")
            return

        assistant.start_microphone_capture()

        logger.info("Voice assistant is running. Press Ctrl+C to stop.")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping voice assistant...")
        finally:
            assistant.stop()
            logger.info("Voice assistant stopped")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        assistant.stop()


if __name__ == "__main__":
    asyncio.run(main())
