# main.py
import os
import asyncio
import logging
import threading
import time
import re
from typing import List, Optional
from queue import Queue


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

# Interrupt / stop words for TTS
INTERRUPT_WORDS = {"stop", "wait", "hold on", "assistant"}

# Modes and defaults
MODE_SETTINGS = {
    "quick": {"BUFFER_MAX_WORDS": 25, "BUFFER_MAX_SECONDS": 0.9, "MIN_INPUT_WINDOW": 0.6},
    "medium": {"BUFFER_MAX_WORDS": 35, "BUFFER_MAX_SECONDS": 1.6, "MIN_INPUT_WINDOW": 1.1},
    "long": {"BUFFER_MAX_WORDS": 60, "BUFFER_MAX_SECONDS": 3.0, "MIN_INPUT_WINDOW": 2.0},
    "manual": {"BUFFER_MAX_WORDS": 9999, "BUFFER_MAX_SECONDS": 9999.0, "MIN_INPUT_WINDOW": 9999.0},
}
DEFAULT_MODE = "quick"

class SpeechBuffer:
    def __init__(self):
        self.text = ""
        self.first_time: Optional[float] = None
        self.last_activity: Optional[float] = None
        self._lock = threading.Lock()

    def append(self, fragment: str):
        now = time.time()
        with self._lock:
            if not self.text:
                self.text = fragment
            else:
                self.text += " " + fragment
            if not self.first_time:
                self.first_time = now
            self.last_activity = now

    def snapshot_and_clear(self) -> str:
        with self._lock:
            s = self.text.strip()
            self.text = ""
            self.first_time = None
            self.last_activity = None
            return s

    def snapshot(self) -> str:
        with self._lock:
            return self.text.strip()

    def time_since_last_activity(self) -> float:
        if not self.last_activity:
            return 1e9
        return time.time() - self.last_activity

    def total_words(self) -> int:
        return len(self.snapshot().split())

class VoiceAssistant:
    def __init__(self, mode: str = DEFAULT_MODE):
        self.deepgram_stt = None
        self.deepgram_connection = None
        self.audio = pyaudio.PyAudio()
        self.event_loop = None

        self.is_listening = False
        self.is_tts_playing = False
        self.stop_tts_flag = False
        self.is_processing = False

        self.buffer = SpeechBuffer()
        self.last_exact_fragment = ""  # simple dedupe
        self.tts_queue = asyncio.Queue()
        self.thoughts_queue = asyncio.Queue()

        # mode
        self.mode = mode if mode in MODE_SETTINGS else DEFAULT_MODE
        self._apply_mode()

        # announce startup once
        logger.info("Quick mode activated. Say 'switch mode medium, switch mode long' if you want long-form speaking.")

        # lock for tts playback
        self._play_lock = threading.Lock()

    def _apply_mode(self):
        s = MODE_SETTINGS[self.mode]
        self.BUFFER_MAX_WORDS = s["BUFFER_MAX_WORDS"]
        self.BUFFER_MAX_SECONDS = s["BUFFER_MAX_SECONDS"]
        self.MIN_INPUT_WINDOW = s["MIN_INPUT_WINDOW"]
        logger.info(f"Mode set to '{self.mode}': words={self.BUFFER_MAX_WORDS}, seconds={self.BUFFER_MAX_SECONDS}, min_window={self.MIN_INPUT_WINDOW}")

    def _handle_switch_mode_command(self, fragment: str) -> bool:
        """Explicit command starts with 'switch mode' (case-insensitive)."""
        t = fragment.lower().strip()
        if not t.startswith("switch mode"):
            return False
        # try to extract 'quick', 'medium', 'long', 'manual' after the phrase
        m = re.search(r"switch mode(?: to)?\s*(quick|medium|long|manual)?", t)
        if m:
            target = m.group(1)
            if target:
                if target in MODE_SETTINGS:
                    self.mode = target
                    self._apply_mode()
                    # speak confirmation
                    asyncio.run_coroutine_threadsafe(self._say_tts_nonblocking(f"Mode switched to {target}."), self.event_loop or asyncio.get_event_loop())
                    return True
        # if ambiguous, ask user (enqueue a short clarification)
        asyncio.run_coroutine_threadsafe(self._say_tts_nonblocking("Which mode would you like: quick, medium, long, or manual?"), self.event_loop or asyncio.get_event_loop())
        return True

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

            def on_message(message: ListenV1SocketClientResponse):
                try:
                    if not isinstance(message, ListenV1ResultsEvent):
                        return
                    channel = getattr(message, "channel", None)
                    if not channel:
                        return
                    alts = getattr(channel, "alternatives", None)
                    if not alts:
                        return
                    alt0 = alts[0]
                    fragment = getattr(alt0, "transcript", "").strip()
                    if not fragment:
                        return

                    # simple exact dedupe
                    if fragment == self.last_exact_fragment:
                        return
                    self.last_exact_fragment = fragment

                    # explicit "switch mode" command — only handled when the phrase starts with that
                    if fragment.lower().startswith("switch mode"):
                        handled = self._handle_switch_mode_command(fragment)
                        if handled:
                            return

                    frag_lower = fragment.lower()
                    # Interrupt handling while TTS is playing
                    if any(w in frag_lower for w in INTERRUPT_WORDS):
                        if self.is_tts_playing and not self.stop_tts_flag:
                            logger.info(f"Interrupt word detected: {fragment}")
                            self.stop_tts_flag = True

                    # Append to internal buffer
                    self.buffer.append(fragment)

                    # If Deepgram signals speech_final, flush immediately
                    speech_final_msg = getattr(message, "speech_final", False)
                    speech_final_alt = getattr(alt0, "speech_final", False)
                    if speech_final_msg or speech_final_alt:
                        # schedule flush on event loop
                        if self.event_loop and self.event_loop.is_running():
                            asyncio.run_coroutine_threadsafe(self._flush_buffer_immediate(), self.event_loop)
                        else:
                            asyncio.create_task(self._flush_buffer_immediate())
                        return

                    # Else: do nothing — buffer_flush_loop will handle fallback/hard-limits
                except Exception as e:
                    logger.error(f"STT message error: {e}")

            def on_error(err):
                logger.error(f"Deepgram STT error: {err}")

            def on_close(_):
                logger.info("Deepgram STT connection closed")
                self.deepgram_stt = None

            self.deepgram_stt.on(EventType.OPEN, lambda _: logger.info("Deepgram STT opened"))
            self.deepgram_stt.on(EventType.MESSAGE, on_message)
            self.deepgram_stt.on(EventType.ERROR, on_error)
            self.deepgram_stt.on(EventType.CLOSE, on_close)

            def listening_thread():
                try:
                    if self.deepgram_stt:
                        self.deepgram_stt.start_listening()
                except Exception as e:
                    logger.error(f"Listening thread error: {e}")

            threading.Thread(target=listening_thread, daemon=True).start()
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

    # ---------------- Immediate flush on speech_final ----------------
    async def _flush_buffer_immediate(self):
        # quick debounce: if processing already, we append to existing handling
        if self.is_processing:
            return
        text = self.buffer.snapshot_and_clear()
        if text:
            await self.process_with_openai(text)

    # ---------------- Buffer flush loop (fallback & hard limit) ----------------
    async def buffer_flush_loop(self):
        poll = 0.25
        while True:
            await asyncio.sleep(poll)
            if self.is_processing:
                continue
            snapshot = self.buffer.snapshot()
            if not snapshot:
                continue
            words = snapshot.split()
            silence = self.buffer.time_since_last_activity()
            enough_time_spoken = False
            if self.buffer.first_time:
                enough_time_spoken = (time.time() - self.buffer.first_time) >= self.MIN_INPUT_WINDOW

            # Hard limit
            if len(words) >= self.BUFFER_MAX_WORDS:
                text = self.buffer.snapshot_and_clear()
                if text:
                    await self.process_with_openai(text)
                continue

            # Pause-based fallback (no speech_final arrived)
            if silence >= self.BUFFER_MAX_SECONDS and enough_time_spoken:
                text = self.buffer.snapshot_and_clear()
                if text:
                    await self.process_with_openai(text)
                continue

    # ---------------- OpenAI processing ----------------
    async def process_with_openai(self, transcript: str):
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
                max_tokens=150,
            )
            response_text = ""
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        response_text += delta.content
                except Exception:
                    pass
            if response_text:
                await self.thoughts_queue.put(response_text)
        except Exception as e:
            logger.error(f"OpenAI processing error: {e}")
        finally:
            self.is_processing = False

    # ---------------- Thoughts worker ----------------
    async def thoughts_worker(self):
        while True:
            thought = await self.thoughts_queue.get()
            await self.tts_queue.put(thought)
            self.thoughts_queue.task_done()

    # ---------------- TTS worker ----------------
    async def tts_worker(self):
        while True:
            text = await self.tts_queue.get()
            await self.stream_to_tts(text)
            self.tts_queue.task_done()

    async def _say_tts_nonblocking(self, text: str):
        """Quick helper to enqueue a short confirmation without blocking caller."""
        await self.tts_queue.put(text)

    async def stream_to_tts(self, text: str):
        """
        Robust TTS: collect incoming audio chunks into a Queue and let a single
        playback thread own the PyAudio stream and write chunks sequentially.
        This prevents write-races (Stream stopped/closed) seen in logs.
        """
        if self.is_tts_playing:
            logger.warning("TTS ignored; already playing")
            return

        logger.info(f"Agent says: {text}")
        self.is_tts_playing = True
        self.stop_tts_flag = False

        try:
            connection = deepgram.speak.v1.connect(
                model="aura-asteria-en",
                encoding="linear16",
                sample_rate=str(SAMPLE_RATE),
            )

            done_event = threading.Event()
            playback_queue: "Queue[bytes]" = Queue(maxsize=200)  # bounded to avoid unbounded memory
            playback_error = {"flag": False}

            def playback_worker():
                """Thread that owns the PyAudio output stream and writes chunks from the queue."""
                out_stream = None
                try:
                    out_stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                                 rate=SAMPLE_RATE, output=True, frames_per_buffer=CHUNK_SIZE)
                    # Keep draining until done_event is set and queue is empty
                    while True:
                        try:
                            # wait for next chunk with a small timeout so we can exit if done_event set
                            chunk = playback_queue.get(timeout=0.3)
                        except Exception:
                            # timeout — check termination condition
                            if done_event.is_set() and playback_queue.empty():
                                break
                            continue

                        if self.stop_tts_flag:
                            # drain and stop
                            logger.info("Playback worker stopping due to interrupt.")
                            break

                        if chunk:
                            try:
                                out_stream.write(chunk)
                            except Exception as e:
                                # Writing failed (stream stopped/closed etc.). Signal and stop.
                                logger.error(f"Playback write error: {e}")
                                playback_error["flag"] = True
                                # signal done and break
                                done_event.set()
                                break
                        playback_queue.task_done()
                except Exception as e:
                    logger.error(f"Playback worker error: {e}")
                    playback_error["flag"] = True
                    done_event.set()
                finally:
                    if out_stream:
                        try:
                            out_stream.stop_stream()
                        except Exception:
                            pass
                        try:
                            out_stream.close()
                        except Exception:
                            pass

            with connection as conn:
                conn.on(EventType.OPEN, lambda _: logger.info("TTS connection opened"))
                conn.on(EventType.ERROR, lambda e: (logger.error(f"TTS error: {e}"), done_event.set()))
                conn.on(EventType.CLOSE, lambda _: logger.info("TTS connection closed"))

                def on_message(msg):
                    try:
                        if isinstance(msg, bytes):
                            # enqueue bytes, but don't block the listener thread indefinitely
                            try:
                                playback_queue.put_nowait(msg)
                            except Exception:
                                # queue full or other problem -> signal error and stop
                                logger.error("Playback queue full or error; stopping TTS.")
                                done_event.set()
                        elif getattr(msg, "type", None) == "done":
                            # mark completion — playback thread will exit after draining queue
                            done_event.set()
                    except Exception as e:
                        logger.error(f"TTS on_message exception: {e}")
                        done_event.set()

                conn.on(EventType.MESSAGE, on_message)

                # start playback thread
                pb_thread = threading.Thread(target=playback_worker, daemon=True)
                pb_thread.start()

                # start listener thread to receive audio messages
                threading.Thread(target=lambda: conn.start_listening(), daemon=True).start()

                # send TTS text
                try:
                    conn.send_text(SpeakV1TextMessage(type="Speak", text=text))
                except Exception as e:
                    logger.error(f"Failed to send TTS text: {e}")
                    done_event.set()

                # Wait for done_event (bounded)
                done_event.wait(timeout=15.0)

                # Wait for playback thread to finish draining queue (bounded)
                pb_thread.join(timeout=5.0)

                # If there was a playback error, log
                if playback_error.get("flag"):
                    logger.error("Playback experienced an error during TTS.")

                # ensure conn close
                try:
                    conn.send_control(SpeakV1ControlMessage(type="Close"))
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
        finally:
            # Clear any leftover queue items to avoid memory growth
            try:
                while not playback_queue.empty():
                    playback_queue.get_nowait()
                    playback_queue.task_done()
            except Exception:
                pass

            self.is_tts_playing = False
            self.stop_tts_flag = False

    # ---------------- Playback helper (not used much now) ----------------
    def play_audio(self, audio_data: bytes):
        # kept for compatibility but current implementation streams chunks directly to output
        try:
            stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                     rate=SAMPLE_RATE, output=True, frames_per_buffer=CHUNK_SIZE)
            bytes_per_frame = 2 * CHANNELS
            step = CHUNK_SIZE * bytes_per_frame
            for i in range(0, len(audio_data), step):
                if self.stop_tts_flag:
                    logger.info("TTS playback interrupted.")
                    break
                chunk = audio_data[i:i + step]
                if chunk:
                    stream.write(chunk)
            stream.stop_stream()
            stream.close()
            logger.info("Audio playback completed (or interrupted)")
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    # ---------------- Stop ----------------
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
        try:
            self.audio.terminate()
        except Exception:
            pass

# ---------------- Main ----------------
async def main():
    assistant = VoiceAssistant(mode=DEFAULT_MODE)

    if not await assistant.connect_deepgram_stt():
        logger.error("Failed to connect to Deepgram STT")
        return

    assistant.start_microphone_capture()

    # background workers
    asyncio.create_task(assistant.buffer_flush_loop())
    asyncio.create_task(assistant.thoughts_worker())
    asyncio.create_task(assistant.tts_worker())

    logger.info("Voice assistant running. Say 'switch mode ...' to change modes.")
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