#Stop TTS immediately if user says a wake/interrupt word.

#Prevent overlapping TTS and “sent 1000” errors.

#Reduce duplicate / filler transcripts triggering OpenAI.

#Keep the system fully async without adding extra delay.

import os
import asyncio
import logging
import threading

import queue
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

# --- Interrupt / wake words for TTS stop ---
INTERRUPT_WORDS = {"stop", "wait", "hold on", "assistant"}  # FIXED

class VoiceAssistant:
    def __init__(self):
        self.deepgram_stt = None
        self.deepgram_connection = None
        self.is_processing = False
        self.current_transcript = ""
        self.event_loop = None
        self.audio = pyaudio.PyAudio()
        self.is_listening = False
        self.is_tts_playing = False
        self.stop_tts_flag = False

    async def connect_deepgram_stt(self):
        try:
            self.event_loop = asyncio.get_event_loop()

            self.deepgram_connection = deepgram.listen.v1.connect(
                model="nova-2",
                language="en-US",

                # Formatting options
                smart_format=True,
                punctuate=True,

                # Audio format
                encoding="linear16",
                channels=str(CHANNELS),
                sample_rate=str(SAMPLE_RATE),

                # Endpointing / VAD
                interim_results=True,
                utterance_end_ms="2800",
                endpointing=None,
                vad_events=False,

                # Optional flags
                profanity_filter=False,
                numerals=False,
                multichannel=False,
                diarize=False
            )

            self.deepgram_stt = self.deepgram_connection.__enter__()

            assistant = self

            def on_message(message: ListenV1SocketClientResponse) -> None:
                try:
                    if isinstance(message, ListenV1ResultsEvent):
                        if hasattr(message, "channel") and message.channel:
                            if hasattr(message.channel, "alternatives") and message.channel.alternatives:

                                sentence = message.channel.alternatives[0].transcript
                                if len(sentence.strip()) == 0:
                                    return
                                sentence_lower = sentence.lower()
                                if any(word in sentence_lower for word in INTERRUPT_WORDS):
                                    if self.is_tts_playing:
                                        logger.info(f"Interrupt word detected: {sentence}")
                                        self.stop_tts_flag = True

                                # --- FIXED: Final flags ---
                                is_final = getattr(message, "is_final", False)
                                speech_final = getattr(message, "speech_final", False)

                                # Accumulate interim fragments
                                if not speech_final:
                                    assistant.current_transcript += " " + sentence
                                    logger.debug(f"Interim: {sentence}")
                                    return

                                # True end of utterance
                                assistant.current_transcript += " " + sentence
                                final_text = assistant.current_transcript.strip()
                                assistant.current_transcript = ""

                                # --- FIXED: ignore too short / filler utterances ---
                                if len(final_text) < 6 or not any(v in final_text.lower() for v in "aeiou"):
                                    logger.debug(f"Ignoring short/filler transcript: {final_text}")
                                    return

                                logger.info(f"Utterance complete: {final_text}")

                                if assistant.event_loop and assistant.event_loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        assistant.process_with_openai(final_text),
                                        assistant.event_loop,
                                    )
                except Exception as e:
                    logger.error(f"Error in on_message handler: {e}")

            def on_error(error):
                logger.error(f"Deepgram STT error: {error}")

            def on_close(_):
                logger.info("Deepgram STT connection closed")
                assistant.deepgram_stt = None

            self.deepgram_stt.on(EventType.OPEN, lambda _: logger.info("Deepgram STT connection opened"))
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

    async def process_with_openai(self, transcript):
        if not transcript.strip() or self.is_processing:
            return

        self.is_processing = True
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
        if self.is_tts_playing:
            logger.warning("TTS request ignored; already playing")
            return
        self.is_tts_playing = True
        self.stop_tts_flag = False

        try:
            connection = deepgram.speak.v1.connect(
                model="aura-asteria-en",
                encoding="linear16",
                sample_rate=str(SAMPLE_RATE),
            )
            audio_chunks = []
            tts_finished = threading.Event()

            with connection as conn:
                conn.on(EventType.OPEN, lambda _: logger.info("TTS connection opened"))
                conn.on(EventType.ERROR, lambda e: (logger.error(f"TTS error: {e}"), tts_finished.set()))
                conn.on(EventType.CLOSE, lambda _: logger.info("TTS connection closed"))

                def on_message(msg):
                    if isinstance(msg, bytes):
                        audio_chunks.append(msg)
                    elif getattr(msg, "type", None) == "done":
                        tts_finished.set()

                conn.on(EventType.MESSAGE, on_message)

                threading.Thread(target=lambda: conn.start_listening(), daemon=True).start()

                # Send TTS
                conn.send_text(SpeakV1TextMessage(type="Speak", text=text))

                # Wait for TTS completion
                tts_finished.wait(timeout=8.0)

                # Playback
                if audio_chunks:
                    self.play_audio(b"".join(audio_chunks))

                conn.send_control(SpeakV1ControlMessage(type="Close"))

        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
        finally:
            self.is_tts_playing = False

    def play_audio(self, audio_data: bytes):
        try:
            stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                     rate=SAMPLE_RATE, output=True, frames_per_buffer=CHUNK_SIZE)
            for i in range(0, len(audio_data), CHUNK_SIZE * 2):
                chunk = audio_data[i:i + CHUNK_SIZE * 2]
                if chunk:
                    stream.write(chunk)
            stream.stop_stream()
            stream.close()
            logger.info("Audio playback completed")
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

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
async def main():
    # Check required environment variables
    required_vars = ["DEEPGRAM_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return

    assistant = VoiceAssistant()

    try:
        # Connect Deepgram STT
        if not await assistant.connect_deepgram_stt():
            logger.error("Failed to connect to Deepgram STT")
            return

        # Start microphone capture
        assistant.start_microphone_capture()

        logger.info("Voice assistant running. Ctrl+C to stop.")

        # Keep the program running
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
