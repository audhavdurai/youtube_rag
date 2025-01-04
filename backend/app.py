from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone_store import PineconeStore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import clip
from datetime import datetime
import torch
import uuid


app = Flask(__name__)
CORS(app)

# Initialize Pinecone store and ChatGPT
class VideoQASystem:
    def __init__(self, openai_api_key, pinecone_api_key):
        self.store = PineconeStore(pinecone_api_key=pinecone_api_key)
        self.chat_model = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            max_tokens=500,
            api_key=openai_api_key
        )
        
    def process_video(self, url, chat_id, frame_interval=60):
        """Process a video URL and store both frames and transcript"""
        try:
            # Store frames
            frames_result = self.store.store_video_frames(url, chat_id, frame_interval=frame_interval)
            
            # Store transcript
            transcript_result = self.store.store_transcript(url, chat_id)
            
            return {
                "status": "success",
                "video_info": frames_result["video_info"],
                "frames_stored": frames_result["frames_stored"],
                "transcript_chunks": transcript_result["chunks_stored"]
            }
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
    
    def _format_prompt(self, data_dict):
        """Format the prompt for ChatGPT with images and text"""
        messages = []
        
        # Add images to the message
        for img_base64 in data_dict["context"]["images"]:
            messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # Format chat history if available
        chat_history = ""
        if "past_messages" in data_dict and data_dict["past_messages"]:
            chat_history = "\nPrevious conversation:\n"
            for msg in data_dict["past_messages"]:
                prefix = "Human: " if msg["type"] == "question" else "Assistant: "
                chat_history += f"{prefix}{msg['content']}\n"
        
        # Add text context and question
        messages.append({
            "type": "text",
            "text": (
                f"You are analyzing a video. Here is the relevant context and question:\n\n"
                f"Transcript excerpt:\n{' '.join(data_dict['context']['texts'])}\n\n"
                f"{chat_history}\n"
                f"Current question: {data_dict['question']}\n\n"
                f"Please answer the current question based on the video content and previous conversation context."
            )
        })


        # "Analyze the video content using the following information:\n"
        #         "1. Visual details from the attached frames from a video\n"
        #         "2. Speech and dialogue from the transcript\n"
        #         "3. Video metadata and context\n\n"
        
        return [HumanMessage(content=messages)]
    
    def query(self, question, chat_id, k=5):
        """Query the video database with a question"""
        try:
            # Get question embedding
            text = clip.tokenize([question]).to(self.store.device)
            with torch.no_grad():
                question_embedding = self.store.model.encode_text(text)
                question_embedding = [float(x) for x in question_embedding.cpu().detach().numpy()[0]]
            
            # Query Pinecone with chat_id filter
            results = self.store.index.query(
                vector=question_embedding,
                top_k=k,
                include_metadata=True,
                filter={"chat_id": {"$eq": chat_id}}
            )

            results_images = self.store.index.query(
                vector=question_embedding,
                top_k=2,
                include_metadata=True,
                filter={
                    "type": {"$eq": "image"},
                    "chat_id": {"$eq": chat_id}
                },
            )

            # Split results into images and texts
            images = []
            texts = []
            for match in results.matches:
                if match.metadata["type"] == "image":
                    images.append(match.metadata["content"])
                else:
                    texts.append(match.metadata["content"])

            for match in results_images.matches:
                if match.metadata["type"] == "image":
                    images.append(match.metadata["content"])
                else:
                    texts.append(match.metadata["content"])
            
            # Create context dictionary
            context = {"images": images, "texts": texts}
            
            past_messages = []
            
            if chat_id in chats:
                chat = chats[chat_id]
                past_messages = chat.messages if chat else []

            # Create and run the chain
            chain = (
                {
                    "context": lambda _: context,
                    "question": RunnablePassthrough(),
                    "past_messages": lambda _: past_messages 
                }
                | RunnableLambda(self._format_prompt)
                | self.chat_model
                | StrOutputParser()
            )
            
            # Get response
            response = chain.invoke(question)
            
            return {
                "answer": response,
                "context": {
                    "images": images[:3],  # Limit to 3 images for response
                    "texts": texts
                }
            }
            
        except Exception as e:
            raise Exception(f"Error querying video content: {str(e)}")

# Initialize the system with API keys
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_6V1rcr_SoVXjwiupo4WysvqJkpDNLVdtjAnjWvn6J66xPsHwQ8xgAzBBBooKzYyahsbajC"
qa_system = VideoQASystem(OPENAI_API_KEY, PINECONE_API_KEY)

chats = {}

class Chat:
    def __init__(self, title=None, username=None):
        self.id = str(uuid.uuid4())
        self.title = title or "New Chat"
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.messages = []
        self.videos = []
        self.username = username

@app.route('/api/chats', methods=['GET'])
def list_chats():
    """Get all chats"""

    username = request.args.get('username')

    return jsonify([{
        'id': chat.id,
        'title': chat.title,
        'created_at': chat.created_at.isoformat(),
        'updated_at': chat.updated_at.isoformat(),
        'last_message': chat.messages[-1] if chat.messages else None, 
        'all_messages': chat.messages, 
        'videos': chat.videos if chat.videos else [], 
    } for chat in chats.values() if chat.username == username])

@app.route('/api/chats', methods=['POST'])
def create_chat():
    """Create a new chat"""
    data = request.get_json()

    print(data)

    chat = Chat(title=data['title'], username = data['username'])
    chats[chat.id] = chat
    return jsonify({
        'id': chat.id,
        'title': chat.title,
        'created_at': chat.created_at.isoformat(),
        'updated_at': chat.updated_at.isoformat(),
        'username': chat.username,
    })

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    """Get a specific chat"""
    chat = chats.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
        
    return jsonify({
        'id': chat.id,
        'title': chat.title,
        'video_url': chat.video_url,
        'created_at': chat.created_at.isoformat(),
        'updated_at': chat.updated_at.isoformat(),
        'messages': chat.messages
    })

@app.route('/api/process-video', methods=['POST'])
def process_video():
    """Process a video URL for a specific chat"""
    try:
        data = request.get_json()
        print(data)
        if not data or 'url' not in data or 'chat_id' not in data:
            return jsonify({"error": "Missing URL or chat_id in request body"}), 400
            
        chat = chats.get(data['chat_id'])
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
            
        # Use the VideoQASystem's process_video method
        result = qa_system.process_video(data['url'], data['chat_id'], frame_interval=int(data['frame_interval']))

        video_info = {
            'url': data['url'],
            'title': result['video_info'].get('title', 'Untitled Video'),
            'timestamp': datetime.utcnow().isoformat()
        }
        chat.videos.append(video_info)
        
        # Update chat
        chat.updated_at = datetime.utcnow()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query():
    """Query video content for a specific chat"""
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'chat_id' not in data:
            return jsonify({"error": "Missing question or chat_id in request body"}), 400
            
        chat = chats[data['chat_id']]
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        print(data)
        # Use the VideoQASystem's query method with chat_id
        result = qa_system.query(data['question'], data['chat_id'])
        
        # Store message in chat history
        chat.messages.append({
            'type': 'question',
            'content': data['question'],
            'timestamp': datetime.utcnow().isoformat()
        })
        chat.messages.append({
            'type': 'answer',
            'content': result['answer'],
            'context': result['context'],
            'timestamp': datetime.utcnow().isoformat()
        })
        chat.updated_at = datetime.utcnow()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Delete a chat and its vectors"""

    print(chat_id)
    chat = chats[chat_id]

    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    # Delete vectors from Pinecone
    # qa_system.store.delete_chat(chat_id)
    
    # Delete chat from memory
    del chats[chat_id]
    
    return jsonify({'status': 'success'})

@app.route('/api/chats/<chat_id>', methods=['PATCH'])
def update_chat(chat_id):
    """Update a chat's title"""
    try:
        data = request.get_json()
        if not data or 'title' not in data:
            return jsonify({"error": "Missing title in request body"}), 400
            
        chat = chats.get(chat_id)
        if not chat:
            return jsonify({"error": "Chat not found"}), 404
        
        chat.title = data['title']
        chat.updated_at = datetime.utcnow()
        
        return jsonify({
            'id': chat.id,
            'title': chat.title,
            'created_at': chat.created_at.isoformat(),
            'updated_at': chat.updated_at.isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)