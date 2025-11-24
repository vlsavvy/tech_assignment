# Deepgram STT/TTS + OpenAI Voice Assistant

A Python application that creates a local desktop voice assistant using:
- **Deepgram STT** for real-time speech-to-text transcription from microphone
- **OpenAI** for processing user queries and generating responses
- **Deepgram TTS** for text-to-speech conversion
- **PyAudio** for microphone input and audio output

## Features

- Real-time voice transcription using Deepgram STT
- Natural language processing with OpenAI GPT-4o-mini
- Text-to-speech streaming using Deepgram TTS (Aura Asteria voice)
- Local microphone input and speaker output
- Asynchronous processing for smooth real-time interaction

## Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager ([Installation guide](https://github.com/astral-sh/uv#installation))
- Deepgram API key ([Get one here](https://deepgram.com))
- OpenAI API key ([Get one here](https://platform.openai.com))
- Microphone and speakers/headphones
- PyAudio system dependencies (see Installation section)

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd tech_assignment
```

2. Install system dependencies for PyAudio:

**On Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**On macOS:**
```bash
brew install portaudio
```

**On Windows:**
PyAudio wheels should be available via pip.

3. Install Python dependencies using uv:
```bash
uv sync
```

This will create a virtual environment and install all dependencies from `pyproject.toml`.

4. Create a `.env` file in the project root:
```bash
touch .env
```

5. Edit `.env` and add your API keys:
```env
DEEPGRAM_API_KEY=your_deepgram_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

## Configuration

### Environment Variables

- `DEEPGRAM_API_KEY`: Your Deepgram API key (required)
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Audio Settings

The application uses the following audio configuration:
- Sample rate: 16kHz
- Format: 16-bit PCM (linear16)
- Channels: Mono (1 channel)
- Chunk size: 1024 frames

## Usage


Create virtual environment. 

```bash
uv venv
```

Activate the virtual environment:
```bash
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

1. Start the voice assistant:
```bash
uv run python main.py
```
`

2. The application will:
   - Connect to Deepgram STT
   - Start listening to your microphone
   - Display "Microphone started. Speak now..." when ready

3. Speak into your microphone. The flow will be:
   - Your speech → Microphone → Deepgram STT → OpenAI → Deepgram TTS → Speakers → You

4. Press `Ctrl+C` to stop the assistant.

## Architecture

```
User Speech (Microphone)
    ↓
PyAudio Capture
    ↓
Deepgram STT (Real-time Transcription)
    ↓
OpenAI API (Process & Generate Response)
    ↓
Deepgram TTS (Text-to-Speech)
    ↓
PyAudio Playback
    ↓
User Hears Response (Speakers)
```

## How It Works

1. **Microphone Capture**: Audio is captured from the default microphone using PyAudio
2. **Deepgram STT**: Audio chunks are sent to Deepgram for real-time transcription
3. **Transcript Processing**: Final transcriptions trigger OpenAI processing
4. **OpenAI Processing**: User queries are sent to OpenAI GPT-4o-mini for response generation
5. **Deepgram TTS**: OpenAI responses are converted to speech using Deepgram TTS (Aura Asteria voice)
6. **Audio Playback**: TTS audio is played through the default speakers/headphones

## Development

The application uses:
- **asyncio** for asynchronous operations
- **threading** for concurrent audio capture and processing
- **Deepgram SDK** for STT and TTS WebSocket connections
- **OpenAI SDK** for language processing
- **PyAudio** for audio I/O


## Task 🏋🏼🏋🏻‍♀️

If you run the application, you will notice it does not handle interruption well. If you talk with pauses AI interrupts you creating a bad user experince. Your task is to do the following:

- Fix this issue. Come up with ways on how you would tackle this issue and fix it writing the code.
- Create a documentation .md file on how you fixed it. 

Fork this repository. Then make changes in the code and send invitation to the following github username as collaborators: @asifrahaman13 @ssomal-fastr

## License

This project is provided as-is for educational and development purposes.

