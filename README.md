# VideoRAG 🚀

VideoRAG is an AI-powered, cross-platform engagement analyzer. It allows creators and marketers to submit a **YouTube Video** alongside an **Instagram Reel** and instantly extracts metadata, calculates engagement benchmarks, transcribes the audio, and lets you chat with the transcripts using a Retrieval-Augmented Generation (RAG) AI Strategist.

![VideoRAG Architecture](frontend/public/logo.png)

## ✨ Key Features
- **Cross-Platform Analytics:** Compares engagement rates, likes, and comments between YouTube and Instagram.
- **Automated Transcription:** Uses `faster-whisper` and `yt-dlp` to automatically download and transcribe spoken audio from both platforms.
- **Intelligent RAG Chatbot:** Powered by Google Gemini 2.5 Flash and LangChain. The Chatbot is strictly context-aware, meaning it will only answer questions based on the specific two videos you are currently analyzing!
- **Persistent Vector Database:** Uses ChromaDB to securely store and vectorize your video transcripts.
- **Modern UI:** Built with React, Tailwind CSS, Framer Motion, and React Router.

## 🛠️ Technology Stack
- **Frontend:** React (Vite), React Router v6, Tailwind CSS, Framer Motion
- **Backend:** Python 3.11, FastAPI, Pydantic v2
- **AI & RAG:** LangChain, Google Generative AI (Gemini 2.5 Flash), HuggingFace Embeddings, ChromaDB
- **Audio Processing:** `yt-dlp`, `faster-whisper`, `ffmpeg`

---

## 💻 Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- [FFmpeg](https://ffmpeg.org/download.html) (Must be installed on your system and added to your PATH)

### 1. Backend Setup
```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the `backend` directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=models/embedding-001
# Or use: EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Start the FastAPI server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
The frontend will be available at `http://localhost:5173`.

---

## 🚀 Production Deployment

This project is fully configured to be deployed on **Vercel** (Frontend) and **Render** (Backend).

### Backend (Render)
Standard Python environments do not come with `ffmpeg` installed. Therefore, a **Dockerfile** is included in the `backend` directory. 
1. Create a New Web Service on Render and connect your GitHub repository.
2. Set the Runtime to **Docker**.
3. Add your `GOOGLE_API_KEY` to the Environment Variables.
4. Deploy!

### Frontend (Vercel)
The frontend utilizes React Router, which requires traffic routing configuration. A `vercel.json` file is included in the `frontend` directory to handle this automatically.
1. Create a New Project on Vercel and connect your GitHub repository.
2. Set the Root Directory to `frontend`.
3. Add an Environment Variable: `VITE_API_URL` pointing to your live Render backend URL (e.g., `https://your-backend.onrender.com/api/v1`).
4. Deploy!

---

## 🏗️ Architecture Notes
- **Routing:** The UI is split into two React Routes (`/` for input, `/results` for the dashboard).
- **Metadata Filtering:** The backend RAG retriever applies a strict ChromaDB `$in` filter matching the active `video_ids`, guaranteeing the AI only talks about the videos currently on your screen.
- **Error Handling:** Gracefully handles missing metrics (like hidden Instagram likes) by evaluating them as `None` and displaying `N/A`, avoiding division-by-zero crashes.
