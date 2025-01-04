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