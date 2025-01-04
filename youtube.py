import os
import yt_dlp
import whisper
import tempfile
from pathlib import Path

class YouTubeProcessor:
    def __init__(self, output_dir=None):
        """
        Initialize the YouTube processor.
        Args:
            output_dir (str, optional): Directory to save files. Defaults to current directory.
        """
        self.output_dir = output_dir or os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        self.processed_files = []
    
    def download_video(self, url):
        """
        Download a YouTube video.
        Args:
            url (str): YouTube video URL
        Returns:
            tuple: (video_path, video_info)
        """
        ydl_opts = {
            'format': 'best[height<=720]',
            'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
            'quiet': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = os.path.join(self.output_dir, ydl.prepare_filename(info))
                
                video_info = {
                    'title': info.get('title'),
                    'uploader': info.get('uploader'),
                    'description': info.get('description'),
                    'duration': info.get('duration'),
                    'view_count': info.get('view_count'),
                    'path': video_path
                }
                
                # Track the video file for later cleanup
                self.processed_files.append(video_path)
                
                return video_path, video_info
                
        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")
    
    def extract_audio(self, video_path):
        """
        Extract audio from video file.
        Args:
            video_path (str): Path to video file
        Returns:
            str: Path to extracted audio file
        """
        try:
            audio_path = os.path.join(self.output_dir, Path(video_path).stem + '.m4a')
            os.system(f'ffmpeg -i "{video_path}" -vn -acodec copy "{audio_path}" -y')
            
            # Track the audio file for later cleanup
            self.processed_files.append(audio_path)
            
            return audio_path
            
        except Exception as e:
            raise Exception(f"Error extracting audio: {str(e)}")
    
    def transcribe(self, audio_path):
        """
        Transcribe audio file using Whisper.
        Args:
            audio_path (str): Path to audio file
        Returns:
            tuple: (transcript_path, transcript_text)
        """
        try:
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            
            # Save transcript to file
            transcript_path = os.path.join(self.output_dir, Path(audio_path).stem + '_transcript.txt')
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            
            # Track the transcript file for later cleanup
            self.processed_files.append(transcript_path)
            
            return transcript_path, result["text"]
            
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")
    
    def process_video(self, url):
        """
        Process a YouTube video: download, extract audio, and transcribe.
        Args:
            url (str): YouTube video URL
        Returns:
            dict: Paths and information about processed files
        """
        try:
            print("Downloading video...")
            video_path, video_info = self.download_video(url)
            
            print("Extracting audio...")
            audio_path = self.extract_audio(video_path)
            
            print("Transcribing audio...")
            transcript_path, transcript_text = self.transcribe(audio_path)
            
            return {
                'video_path': video_path,
                'audio_path': audio_path,
                'transcript_path': transcript_path,
                'video_info': video_info,
                'transcript': transcript_text
            }
            
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
    
    def cleanup(self):
        """
        Remove all processed files and temporary directory
        """
        try:
            # Remove all tracked files
            for file_path in self.processed_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove temporary directory
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
            
            # Reset processed files list
            self.processed_files = []
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")