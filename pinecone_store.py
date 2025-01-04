import cv2
import base64
import uuid
import clip
import torch
from PIL import Image
import io
from pinecone import Pinecone, ServerlessSpec
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from youtube import YouTubeProcessor

class PineconeStore:
    def __init__(self, pinecone_api_key, index_name="youtube-content"):
        """Initialize Pinecone storage with CLIP embeddings"""
        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Keep OpenCLIP for text embeddings
        self.text_embeddings = OpenCLIPEmbeddings()
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=512,  # CLIP ViT-B/32 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        self.index = self.pc.Index(index_name)
        self.youtube_processor = YouTubeProcessor()

    def _base64_to_pil(self, base64_str):
        """Convert base64 string to PIL Image"""
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    
    def _generate_image_embedding(self, image):
        """Generate CLIP embedding for an image"""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
            embedding = embedding.cpu().numpy()
            
        return embedding[0].tolist()

    def _extract_frames(self, video_path, interval=60):
        """Extract frames from video at given interval"""
        frames = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps * interval)
        
        frame_count = 0
        success = True
        while success:
            success, frame = video.read()
            if frame_count % frame_skip == 0 and success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small_frame = cv2.resize(frame_rgb, (200, int(200 * frame_rgb.shape[0] / frame_rgb.shape[1])))
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                _, buffer = cv2.imencode('.jpg', small_frame, encode_params)
                frames.append(base64.b64encode(buffer).decode('utf-8'))
            frame_count += 1
            
        video.release()
        return frames, frame_count

    def store_video_frames(self, youtube_url: str, chat_id: str, frame_interval=60):
        """
        Download YouTube video and store frames in Pinecone with chat_id
        Args:
            youtube_url (str): YouTube video URL
            chat_id (str): Unique identifier for the chat
            frame_interval (int): Interval between frames in seconds
        """
        try:
            video_path, video_info = self.youtube_processor.download_video(youtube_url)
            
            print("Extracting frames...")
            frames, frame_count = self._extract_frames(video_path, frame_interval)
            print(f"Extracted {len(frames)} frames")
            
            print("Storing frames in Pinecone...")
            frame_ids = []
            vectors_to_upsert = []
            
            for i, frame in enumerate(frames):
                if len(frame) > 30000:
                    print(f"Warning: Frame {i} size {len(frame)} bytes is too large, skipping...")
                    continue
                
                pil_image = self._base64_to_pil(frame)
                embedding = self._generate_image_embedding(pil_image)
                vector_id = f"{chat_id}_frame_{str(uuid.uuid4())}"
                frame_ids.append(vector_id)
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "content": frame,
                        "type": "image",
                        "chat_id": chat_id,
                        "video_id": youtube_url,
                        "video_title": video_info['title'],
                        "timestamp": frame_count * frame_interval,
                        "frame_number": i
                    }
                })
                
                # Batch upsert every 100 vectors
                if len(vectors_to_upsert) >= 100:
                    self.index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
            
            # Upsert any remaining vectors
            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert)
            
            self.youtube_processor.cleanup()

            return {
                "video_info": video_info,
                "frames_stored": len(frames),
                "frame_ids": frame_ids
            }
            
        except Exception as e:
            raise Exception(f"Error storing video frames: {str(e)}")

    def _chunk_text(self, text, chunk_size=250, overlap=20):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                last_period = text.rfind('.', start, end)
                last_space = text.rfind(' ', start, end)
                break_point = max(last_period, last_space)
                if break_point > start:
                    end = break_point + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < text_len else end
            
        return chunks

    def store_transcript(self, youtube_url: str, chat_id: str):
        """
        Store transcript chunks in Pinecone with chat_id
        Args:
            youtube_url (str): YouTube video URL
            chat_id (str): Unique identifier for the chat
        """
        try:
            result = self.youtube_processor.process_video(youtube_url)
            transcript = result['transcript']
            
            print("Chunking transcript...")
            transcript_chunks = self._chunk_text(transcript)
            print(f"Created {len(transcript_chunks)} chunks")
            
            print("Storing transcript chunks in Pinecone...")
            chunk_ids = []
            vectors_to_upsert = []
            
            for i, chunk in enumerate(transcript_chunks):
                text = clip.tokenize([chunk]).to(self.device)
                with torch.no_grad():
                    embedding = self.model.encode_text(text)
                    embedding = [float(x) for x in embedding.cpu().detach().numpy()[0]]
                
                vector_id = f"{chat_id}_transcript_{str(uuid.uuid4())}"
                chunk_ids.append(vector_id)
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "content": f"Transcript chunk {i+1}: {chunk}",
                        "type": "text",
                        "chat_id": chat_id,
                        "content_type": "transcript",
                        "chunk_number": i,
                        "video_id": youtube_url,
                        "video_title": result['video_info']['title']
                    }
                })
                
                # Batch upsert every 100 vectors
                if len(vectors_to_upsert) >= 100:
                    self.index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
            
            # Upsert any remaining vectors
            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert)
            
            self.youtube_processor.cleanup()
            
            return {
                "video_info": result['video_info'],
                "transcript_path": result['transcript_path'],
                "chunks_stored": len(chunk_ids),
                "chunk_ids": chunk_ids
            }
            
        except Exception as e:
            raise Exception(f"Error storing transcript: {str(e)}")

    def query_chat(self, question_embedding: list, chat_id: str, k: int = 5):
        """
        Query vectors for a specific chat
        Args:
            question_embedding: The embedding of the question
            chat_id: The chat to query within
            k: Number of results to return
        """
        try:
            results = self.index.query(
                vector=question_embedding,
                top_k=k,
                include_metadata=True,
                filter={
                    "chat_id": {"$eq": chat_id}
                }
            )
            return results
        except Exception as e:
            raise Exception(f"Error querying vectors: {str(e)}")

    def delete_chat(self, chat_id: str):
        """
        Delete all vectors associated with a chat
        Args:
            chat_id: The chat to delete
        """
        try:
            self.index.delete(
                filter={
                    "chat_id": {"$eq": chat_id}
                }
            )
            return {"status": "success"}
        except Exception as e:
            raise Exception(f"Error deleting chat vectors: {str(e)}")