# ThinkBro 🧠

> Your personal AI trainer that learns from thousands of hours of fitness content, remembers what matters, and answers your questions — grounded in real transcripts, quotes, and wisdom.

From *Sam Sulek's mindset* to *Huberman's protocols* to your coach's advice — ThinkBro **retrieves** what's relevant and **responds** like a knowledgeable personal trainer who's watched it all.

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-prerequisites">Prerequisites</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a>
</p>

## ✨ Features

- 🎙️ **Smart Transcription**: Convert YouTube videos to text using `yt-dlp` + `faster-whisper`
- ✂️ **Audio Enhancement**: Auto-trim silence with `ffmpeg` for cleaner processing
- 🔊 **Advanced Embedding**: Transform speech into vectors using `sentence-transformers`
- 🗃️ **Robust Storage**: Store metadata in PostgreSQL and vectors in Chroma DB
- 🔍 **Intelligent Search**: Semantic search and contextual retrieval of transcripts
- 🤖 **Local AI**: Powered by `Ollama` for privacy-focused, grounded responses
- 🔛 **Versatile Learning**: Compatible with fitness, education, podcasts, and more
- 🔐 **Privacy First**: 100% local, self-hosted, zero API keys needed

## 🛠 Tech Stack

| Component        | Technology               | Purpose                                |
|-----------------|-------------------------|----------------------------------------|
| Video Download  | `yt-dlp`               | Fetch content from YouTube             |
| Audio Processing| `ffmpeg`               | Clean and optimize audio               |
| Transcription   | `faster-whisper`       | Convert speech to text                 |
| Embeddings      | `sentence-transformers`| Generate semantic vectors              |
| Vector Storage  | `Chroma`               | Store and query embeddings             |
| Metadata DB     | `PostgreSQL`           | Manage structured data                 |
| AI Engine       | `Ollama`               | Local language model processing        |

## 📋 Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Ollama installed and running
- FFmpeg

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/think-bro.git
   cd think-bro
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start required services:
   ```bash
   docker-compose up -d  # Starts PostgreSQL and Chroma
   ollama serve         # Start Ollama in a separate terminal
   ```

## 💡 Usage

1. Start ThinkBro:
   ```bash
   python main.py
   ```

2. Ask questions naturally:
   ```
   > what's sam's advice on warming up for legs?

   🗣️ ThinkBro says:
   "In one of his bulk day episodes, Sam mentions starting light to feel the movement, 
   not to impress. It's not about the weight — it's about control and tension."
   ```

### Example Questions

- "What does Sam Sulek think about creatine?"
- "How should I approach progressive overload?"
- "What's Huberman's take on pre-workout supplements?"
- "Share some motivational quotes about pushing through plateaus"

## 🙏 Acknowledgments

- Thanks to all the content creators whose wisdom this project helps organize
- The amazing open-source communities behind the tools we use

---

<p align="center">
Made with ❤️ for the fitness community
</p>