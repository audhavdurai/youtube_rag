# Video QA System

A video question-answering system that allows users to search for and process Youtube videos through natural language queries. The system processes video content and provides intelligent responses to user questions about the video content.

## Features

- **YouTube Video Processing**: Process YouTube videos with link and ask questions about content
- **Visual Context**: View relevant video frames alongside answers for better understanding
- **YouTube Search**: Built-in YouTube video search functionality
- **Chat Management**: Organize conversations in separate chat sessions with messages saved across sessions
- **Multi-user Support**: Username-based login and chat organization

## Technology Stack

### Frontend
- React 
- Tailwind CSS for styling
- Lucide React for icons
- Embedded Youtube videos with iframe
- API calls to Flask

### Backend
- Flask
- yt-dlp for YouTube video processing and search
- OpenAI Whisper for speech-to-text
- CLIP for multimodal text and visual embeddings
- Pinecone for vector storage
- Langchain and GPT-4 for response generation

## Prerequisites

- Python
- Node.js
- OpenAI API key
- Pinecone API key

## Installation & Setup

### Frontend
```bash
# Navigate to frontend directory
cd video-qa

# Install dependencies
npm install

# Start development server
npm start

# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux 
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create .env file with your API keys
echo "OPENAI_API_KEY=your_openai_key" > .env
echo "PINECONE_API_KEY=your_pinecone_key" >> .env
echo "PINECONE_ENVIRONMENT=your_pinecone_environment" >> .env

# Start Flask server
python app.py
