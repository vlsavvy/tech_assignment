import os
import asyncio
import logging
import threading
import time

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV1SocketClientResponse,
    SpeakV1TextMessage,
    SpeakV1ControlMessage,
)
from deepgram.extensions.types.sockets.listen_v1_results_event import ListenV1ResultsEvent
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

if not DEEPGRAM_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing required API keys")

deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

INTERRUPT_WORDS = {"stop", "wait", "hold on", "assistant"}

# --- Improved pause handling ---
BUFFER_MAX_WORDS = 35           # previously 20
BUFFER_MAX_SECONDS = 2.2        # previously 1.5
MIN_INPUT_WINDOW = 1.1          # ensures user speaks enough before reply

class VoiceAssistant:
    def __init__(self):
        self.deepgram_stt = None
        self.deepgram_connection = None
        self.audio = pyaudio.PyAudio()
        self.event_loop = None
        self.is_listening = False
        self.is_tts_playing = False
        self.stop_tts_flag = False
        self.is_processing = False

        self.speech_buffer = ""
        self.last_buffer_time = time.time()
        self.first_buffer_time = None

        self.tts_queue = asyncio.Queue()

    # ---------------- STT Connection ----------------
    async def connect_deepgram_stt(self):
        try:
            self.event_loop = asyncio.get_event_loop()
            self.deepgram_connection = deepgram.listen.v1.connect(
                model="nova-2",
                language="en-US",
                smart_format=True,
                punctuate=True,
                encoding="linear16",
                channels=str(CHANNELS),
                sample_rate=str(SAMPLE_RATE),
                interim_results=True,
                utterance_end_ms="2800",
                endpointing=None,
                vad_events=False,
                profanity_filter=False,
                numerals=False,
                multichannel=False,
                diarize=False
            )
            self.deepgram_stt = self.deepgram_connection.__enter__()

            assistant = self

            def on_message(message: ListenV1SocketClientResponse):
                try:
                    if isinstance(message, ListenV1ResultsEvent):
                        if getattr(message, "channel", None) and getattr(message.channel, "alternatives", None):
                            sentence = message.channel.alternatives[0].transcript.strip()
                            if not sentence:
                                return

                            sentence_lower = sentence.lower()

                            # Interrupt TTS
                            if any(word in sentence_lower for word in INTERRUPT_WORDS):
                                if self.is_tts_playing:
                                    self.stop_tts_flag = True

                            # Buffer logic
                            if not self.first_buffer_time:
                                self.first_buffer_time = time.time()

                            self.speech_buffer += " " + sentence
                            self.last_buffer_time = time.time()

                except Exception as e:
                    logger.error(f"STT message error: {e}")

            self.deepgram_stt.on(EventType.OPEN, lambda _: logger.info("Deepgram STT opened"))
            self.deepgram_stt.on(EventType.MESSAGE, on_message)
            self.deepgram_stt.on(EventType.ERROR, lambda e: logger.error(f"STT error: {e}"))
            self.deepgram_stt.on(EventType.CLOSE, lambda _: logger.info("Deepgram STT closed"))

            def listen_thread():
                try:
                    if self.deepgram_stt:
                        self.deepgram_stt.start_listening()
                except Exception as e:
                    logger.error(f"Listening thread error: {e}")

            threading.Thread(target=listen_thread, daemon=True).start()
            await asyncio.sleep(0.5)
            logger.info("Deepgram STT connected")
            return True
        except Exception as e:
            logger.error(f"STT connection failed: {e}")
            return False

    # ---------------- Microphone Capture ----------------
    def start_microphone_capture(self):
        self.is_listening = True

        def capture_thread():
            try:
                stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                         rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
                logger.info("Microphone started. Speak now...")

                while self.is_listening:
                    try:
                        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        if self.deepgram_stt:
                            self.deepgram_stt.send_media(data)
                    except Exception as e:
                        if "closed" not in str(e).lower():
                            logger.error(f"Microphone send error: {e}")
                        break

                stream.stop_stream()
                stream.close()

            except Exception as e:
                logger.error(f"Microphone thread error: {e}")

        threading.Thread(target=capture_thread, daemon=True).start()

    # ---------------- Buffer Flush Loop ----------------
    async def buffer_flush_loop(self):
        while True:
            await asyncio.sleep(0.3)

            if not self.speech_buffer.strip() or self.is_processing:
                continue

            time_since_last = time.time() - self.last_buffer_time
            words = self.speech_buffer.split()

            enough_time_spoken = (time.time() - self.first_buffer_time) >= MIN_INPUT_WINDOW

            if len(words) >= BUFFER_MAX_WORDS or (time_since_last >= BUFFER_MAX_SECONDS and enough_time_spoken):
                text_to_process = self.speech_buffer.strip()
                self.speech_buffer = ""
                self.first_buffer_time = None
                await self.process_with_openai(text_to_process)

    # ---------------- OpenAI Processing ----------------
    async def process_with_openai(self, transcript):
        if not transcript.strip():
            return

        self.is_processing = True
        try:
            logger.info(f"Processing with OpenAI: {transcript}")

            stream = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for voice interaction."},
                    {"role": "user", "content": transcript}
                ],
                stream=True,
                temperature=0.7,
                max_tokens=150
            )

            response_text = ""

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content

            if response_text:
                logger.info("OpenAI produced thought enqueued for TTS")
                await self.tts_queue.put(response_text)

        except Exception as e:
            logger.error(f"OpenAI processing error: {e}")
        finally:
            self.is_processing = False

    # ---------------- TTS Worker ----------------
    async def tts_worker(self):
        while True:
            text = await self.tts_queue.get()
            await self.stream_to_tts(text)
            self.tts_queue.task_done()

    async def stream_to_tts(self, text):
        if self.is_tts_playing:
            logger.warning("TTS ignored; already playing")
            return

        # 🔥 PRINT WHAT AGENT SPEAKS
        logger.info(f"Agent says: {text}")

        self.is_tts_playing = True
        self.stop_tts_flag = False

        try:
            connection = deepgram.speak.v1.connect(
                model="aura-asteria-en",
                encoding="linear16",
                sample_rate=str(SAMPLE_RATE)
            )

            audio_chunks = []
            tts_finished = threading.Event()

            with connection as conn:
                conn.on(EventType.OPEN, lambda _: logger.info("TTS connection opened"))
                conn.on(EventType.ERROR, lambda e: (logger.error(f"TTS error: {e}"), tts_finished.set()))
                conn.on(EventType.CLOSE, lambda _: logger.info("TTS connection closed"))

                def on_message(msg):
                    if isinstance(msg, bytes):
                        if self.stop_tts_flag:
                            tts_finished.set()
                        else:
                            audio_chunks.append(msg)
                    elif getattr(msg, "type", None) == "done":
                        tts_finished.set()

                conn.on(EventType.MESSAGE, on_message)

                threading.Thread(target=lambda: conn.start_listening(), daemon=True).start()
                conn.send_text(SpeakV1TextMessage(type="Speak", text=text))

                tts_finished.wait(timeout=8.0)

                if audio_chunks and not self.stop_tts_flag:
                    self.play_audio(b"".join(audio_chunks))

                conn.send_control(SpeakV1ControlMessage(type="Close"))

        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
        finally:
            self.is_tts_playing = False

    # ---------------- Audio Playback ----------------
    def play_audio(self, audio_data: bytes):
        try:
            stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                     rate=SAMPLE_RATE, output=True, frames_per_buffer=CHUNK_SIZE)
            for i in range(0, len(audio_data), CHUNK_SIZE * 2):
                stream.write(audio_data[i:i + CHUNK_SIZE * 2])

            logger.info("Audio playback completed (or interrupted)")
            stream.stop_stream()
            stream.close()

        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    # ---------------- Stop Everything ----------------
    def stop(self):
        self.is_listening = False

        if self.deepgram_connection:
            try:
                self.deepgram_connection.__exit__(None, None, None)
            except Exception:
                pass
            finally:
                self.deepgram_stt = None
                self.deepgram_connection = None

        self.audio.terminate()

# ---------------- Main ----------------
async def main():
    assistant = VoiceAssistant()

    if not await assistant.connect_deepgram_stt():
        logger.error("Failed to connect to Deepgram STT")
        return

    assistant.start_microphone_capture()

    asyncio.create_task(assistant.tts_worker())
    asyncio.create_task(assistant.buffer_flush_loop())

    logger.info("Voice assistant running. Ctrl+C to stop.")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping voice assistant...")
    finally:
        assistant.stop()
        logger.info("Voice assistant stopped")

if __name__ == "__main__":
    asyncio.run(main())
