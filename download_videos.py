import os
import subprocess
import argparse
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
import random
from datetime import datetime
import signal
import sys
import shutil
import re
from urllib.parse import urlparse, unquote
from audio_processor import trim_silence
from transcribe import TranscriptionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_progress.log'),
        logging.StreamHandler()
    ]
)

class DownloadManager:
    def __init__(self, channel_url, output_dir, cookies_file=None, browser_cookies=None, max_workers=3, rate_limit=2):
        self.channel_url = channel_url
        self.base_dir = Path(output_dir)
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.cookies_file = cookies_file
        self.browser_cookies = browser_cookies
        
        if cookies_file and browser_cookies:
            logging.warning("Both cookies file and browser cookies specified. Using cookies file.")
            self.browser_cookies = None

        # Initialize channel-specific paths after getting channel info
        self.channel_dir = None
        self.audio_dir = None
        self.metadata_dir = None
        self.channel_info_dir = None
        self.transcripts_dir = None
        self.archive_file = None
        self.failed_downloads = None
        
        # Initialize transcription manager
        self.transcriber = TranscriptionManager(
            model_size="medium",
            device="cpu",
            compute_type="int8",
            cpu_threads=12,
            num_workers=4 
        )

    def extract_channel_name_from_url(self, url):
        """Extract channel name from different YouTube URL formats"""
        parsed = urlparse(url)
        path = unquote(parsed.path)  # Handle URL-encoded characters
        
        # Handle different URL patterns
        patterns = [
            r'/@([^/]+)',           # /@username
            r'/c/([^/]+)',          # /c/channelname
            r'/user/([^/]+)',       # /user/username
            r'/channel/([^/]+)',    # /channel/id
        ]
        
        for pattern in patterns:
            match = re.search(pattern, path)
            if match:
                return match.group(1)
        
        # If no pattern matches, try to get the last part of the path
        parts = [p for p in path.split('/') if p]
        return parts[-1] if parts else 'unknown_channel'
        
    def setup_directories(self, channel_id, channel_title):
        """Create organized directory structure for the channel"""
        # Get channel name from URL first
        url_channel_name = self.extract_channel_name_from_url(self.channel_url)
        
        # Create sanitized channel name for filesystem
        safe_channel_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                                  for c in url_channel_name).strip()
        
        # Add channel ID as suffix for uniqueness
        safe_channel_name = f"{safe_channel_name}_{channel_id}"
        
        # Setup directory structure
        self.channel_dir = self.base_dir / safe_channel_name
        self.audio_dir = self.channel_dir / "audio"
        self.metadata_dir = self.channel_dir / "metadata"
        self.channel_info_dir = self.channel_dir / "channel_info"
        self.transcripts_dir = self.channel_dir / "transcriptions"
        
        # Create all directories
        self.channel_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.channel_info_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup tracking files
        self.archive_file = self.channel_info_dir / 'downloaded_videos.txt'
        self.failed_downloads = self.channel_info_dir / 'failed_downloads.txt'
        
        if not self.archive_file.exists():
            self.archive_file.touch()
        if not self.failed_downloads.exists():
            self.failed_downloads.touch()

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        logging.info("Received shutdown signal. Cleaning up...")
        sys.exit(0)

    def rate_limit_wait(self):
        """Implement rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _add_cookie_args(self, command):
        """Helper method to add appropriate cookie arguments to yt-dlp command"""
        if self.cookies_file:
            command.extend(["--cookies", str(self.cookies_file)])
        elif self.browser_cookies:
            command.extend(["--cookies-from-browser", self.browser_cookies])
        return command

    def get_channel_info(self):
        """Get total video count and channel metadata"""
        try:
            command = [
                "./yt-dlp",
                "--dump-single-json",
                "--flat-playlist",
            ]
            
            # Add cookie arguments
            command = self._add_cookie_args(command)
            command.append(self.channel_url)
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                entries = data.get('entries', [])
                channel_id = data.get('channel_id', 'unknown')
                channel_title = data.get('title', 'Unknown Channel')
                
                # Setup directory structure now that we have channel info
                self.setup_directories(channel_id, channel_title)
                
                # Extract relevant channel metadata
                channel_metadata = {
                    'title': channel_title,
                    'channel_id': channel_id,
                    'channel_url': data.get('channel_url', ''),
                    'description': data.get('description', ''),
                    'video_count': len(entries),
                    'playlist_url': self.channel_url,
                    'extracted_date': datetime.now().isoformat()
                }
                
                # Save channel metadata
                with open(self.channel_info_dir / 'channel_metadata.json', 'w') as f:
                    json.dump(channel_metadata, f, indent=4)
                
                return entries, channel_title, len(entries)
            else:
                error_msg = f"yt-dlp error (code {result.returncode}):\n"
                if result.stderr:
                    error_msg += f"stderr: {result.stderr}\n"
                if result.stdout:
                    error_msg += f"stdout: {result.stdout}"
                logging.error(error_msg)
                raise RuntimeError(f"Failed to get channel info: {error_msg}")
            
        except json.JSONDecodeError as e:
            error_msg = "Failed to parse yt-dlp output. This might indicate a network issue or invalid channel URL."
            logging.error(f"{error_msg}\nError: {str(e)}")
            raise RuntimeError(error_msg)
        except Exception as e:
            logging.error(f"Error getting channel info: {str(e)}")
            raise

    def prompt_for_video_count(self, total_videos, existing_count):
        """Interactive prompt to select number of videos to download"""
        print("\n=== Channel Download Options ===")
        print(f"Total videos available: {total_videos}")
        print(f"Already downloaded: {existing_count}")
        print(f"New videos available: {total_videos - existing_count}")
        print("\nOptions:")
        print("1. Download all new videos")
        print("2. Download specific number of oldest videos")
        print("3. Download specific number of latest videos")
        print("4. Download specific video by ID")
        print("5. Cancel download")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-5): ")
                # Strip whitespace and handle empty input
                choice = choice.strip()
                if not choice:
                    continue
                    
                if choice == '5':
                    print("Download cancelled.")
                    sys.exit(0)
                elif choice == '4':
                    video_id = input("\nEnter the YouTube video ID: ").strip()
                    if video_id:
                        return -1, False, video_id  # Special case for single video
                elif choice in ['1', '2', '3']:
                    if choice == '1':
                        return total_videos - existing_count, True, None  # True for newest first
                    else:
                        while True:
                            try:
                                num_input = input(f"\nHow many videos do you want to download? (1-{total_videos}): ")
                                # Handle empty input
                                if not num_input.strip():
                                    continue
                                    
                                num = int(num_input)
                                if 1 <= num <= total_videos:
                                    return num, (choice == '3'), None  # Return number and whether to get newest first
                                print(f"Please enter a number between 1 and {total_videos}")
                            except ValueError:
                                print("Please enter a valid number")
                            except KeyboardInterrupt:
                                print("\nDownload cancelled.")
                                sys.exit(0)
                else:
                    print("Please enter a valid choice (1-5)")
            except KeyboardInterrupt:
                print("\nDownload cancelled.")
                sys.exit(0)

    def get_video_metadata(self, video_id):
        """Extract useful metadata for a single video"""
        try:
            command = [
                "./yt-dlp",
                "--dump-single-json",
                "--no-playlist",
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                full_data = json.loads(result.stdout)
                # Extract only the most relevant metadata for LLM processing
                useful_metadata = {
                    'id': video_id,
                    'title': full_data.get('title', ''),
                    'description': full_data.get('description', ''),
                    'upload_date': full_data.get('upload_date', ''),
                    'duration': full_data.get('duration', 0),
                    'view_count': full_data.get('view_count', 0),
                    'like_count': full_data.get('like_count', 0),
                    'tags': full_data.get('tags', []),
                    'categories': full_data.get('categories', []),
                    'language': full_data.get('language', ''),
                    'audio_language': full_data.get('audio_language', ''),
                    'automatic_captions': bool(full_data.get('automatic_captions', {})),
                    'subtitles': bool(full_data.get('subtitles', {})),
                    'chapters': full_data.get('chapters', []),
                    'extracted_date': datetime.now().isoformat()
                }
                
                # Save metadata to the channel's metadata directory
                metadata_path = self.metadata_dir / f"{video_id}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(useful_metadata, f, indent=4)
                return useful_metadata
            return None
        except Exception as e:
            logging.error(f"Error getting metadata for video {video_id}: {str(e)}")
            return None

    def download_video(self, video_id):
        """Download a single video with retries and metadata extraction"""
        max_retries = 3
        last_error = None
        
        try:
            metadata = self.get_video_metadata(video_id)
        except Exception as e:
            logging.error(f"Failed to get metadata for video {video_id}: {str(e)}")
            metadata = None
        
        for attempt in range(max_retries):
            try:
                self.rate_limit_wait()
                
                # Get initial list of WAV files
                initial_wav_files = set(self.audio_dir.glob("*.wav"))
                
                # Use video ID for filename
                filename_template = f"{video_id}.%(ext)s"
                
                command = [
                    "./yt-dlp",
                    "--extract-audio",
                    "--audio-format", "wav",
                    "--audio-quality", "0",
                    "-o", str(self.audio_dir / filename_template),
                    "--no-warnings",
                    "--ignore-errors",
                    "--verbose",  # Add verbose output for debugging
                ]
                
                # Add cookie arguments
                command = self._add_cookie_args(command)
                command.append(f"https://www.youtube.com/watch?v={video_id}")
                
                logging.info(f"Attempting to download video {video_id} (attempt {attempt + 1}/{max_retries})")
                result = subprocess.run(command, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Get new list of WAV files
                    current_wav_files = set(self.audio_dir.glob("*.wav"))
                    # Find newly added WAV files
                    new_wav_files = list(current_wav_files - initial_wav_files)
                    
                    if new_wav_files:
                        # Use the most recently created WAV file
                        audio_file = max(new_wav_files, key=lambda p: p.stat().st_mtime)
                        logging.info(f"Found new WAV file: {audio_file.name}")
                        
                        try:
                            # Process the audio file to remove silence
                            trim_silence(audio_file)
                            logging.info(f"Successfully processed audio for video {video_id}")
                            
                            # Transcribe the processed audio
                            try:
                                self.transcriber.transcribe_audio(
                                    audio_file,
                                    output_dir=self.transcripts_dir,
                                    output_filename=video_id  # Use video ID for transcription filename
                                )
                                logging.info(f"Successfully transcribed video {video_id}")
                            except Exception as e:
                                logging.error(f"Error transcribing video {video_id}: {str(e)}")
                                # Continue with the download even if transcription fails
                                
                        except Exception as e:
                            logging.error(f"Error processing audio for video {video_id}: {str(e)}")
                            # Continue with the download even if processing fails
                            
                        # Add to archive after successful download and processing
                        with open(self.archive_file, 'a') as f:
                            f.write(f"youtube {video_id}\n")
                            
                        return True, video_id
                    else:
                        error_msg = (
                            f"No new WAV file found after download for video {video_id}. "
                            f"yt-dlp output: {result.stdout}"
                        )
                        logging.error(error_msg)
                        last_error = error_msg
                else:
                    error_msg = f"yt-dlp error (code {result.returncode}) for video {video_id}:\n"
                    if result.stderr:
                        error_msg += f"stderr: {result.stderr}\n"
                    if result.stdout:
                        error_msg += f"stdout: {result.stdout}"
                    logging.error(error_msg)
                    last_error = error_msg
                
                if attempt < max_retries - 1:
                    wait_time = random.uniform(1, 5) * (attempt + 1)
                    logging.info(f"Retrying video {video_id} in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                error_msg = f"Error downloading video {video_id}: {str(e)}"
                logging.error(error_msg)
                last_error = error_msg
                if attempt < max_retries - 1:
                    wait_time = random.uniform(1, 5) * (attempt + 1)
                    logging.info(f"Retrying video {video_id} in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
        
        # If we get here, all attempts failed
        with open(self.failed_downloads, 'a') as f:
            f.write(f"{video_id}\t{datetime.now().isoformat()}\t{last_error}\n")
        return False, video_id

    def download_single_video(self, video_id):
        """Download a specific video and maintain the channel structure"""
        try:
            # First verify the video exists and get its channel info
            command = [
                "./yt-dlp",
                "--dump-single-json",
                "--no-playlist",
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            command = self._add_cookie_args(command)
            
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                video_data = json.loads(result.stdout)
                channel_id = video_data.get('channel_id', 'unknown')
                channel_title = video_data.get('channel', 'Unknown Channel')
                
                # Setup directory structure if it doesn't exist
                if not self.channel_dir:
                    self.setup_directories(channel_id, channel_title)
                
                # Check if video is already downloaded
                if self.archive_file.exists():
                    with open(self.archive_file, 'r') as f:
                        if any(video_id in line for line in f):
                            logging.info(f"Video {video_id} already downloaded")
                            return True
                
                # Download the video
                success, _ = self.download_video(video_id)
                return success
            else:
                logging.error(f"Error getting video info: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error downloading single video {video_id}: {str(e)}")
            return False

    def download_channel(self):
        """Main method to handle channel download"""
        videos, channel_title, total_videos = self.get_channel_info()
        if not videos:
            error_msg = "No videos found or error getting channel info"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        logging.info(f"Starting download for channel: {channel_title}")
        logging.info(f"Total videos found: {len(videos)}")

        # Filter out already downloaded videos
        downloaded_videos = set()
        if self.archive_file.exists():
            with open(self.archive_file, 'r') as f:
                downloaded_videos = set(line.strip().split()[1] for line in f)

        # Get user's download preference
        num_videos, newest_first, specific_video = self.prompt_for_video_count(total_videos, len(downloaded_videos))
        
        # Handle single video download
        if specific_video:
            success = self.download_single_video(specific_video)
            if success:
                logging.info(f"Successfully downloaded video {specific_video}")
            else:
                logging.error(f"Failed to download video {specific_video}")
            return
        
        # Filter and sort videos
        available_videos = [v for v in videos if v['id'] not in downloaded_videos]
        
        if not available_videos:
            logging.info("No new videos to download")
            return
        
        # Sort videos based on user preference (newest or oldest first)
        if newest_first:
            available_videos = available_videos[::-1]  # Reverse to get newest first
            
        # Limit to requested number
        videos_to_download = [v['id'] for v in available_videos[:num_videos]]
        
        logging.info(f"Preparing to download {len(videos_to_download)} videos")
        
        successful_downloads = 0
        failed_downloads = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_video, video_id): video_id 
                      for video_id in videos_to_download}
            
            with tqdm(total=len(videos_to_download), desc="Downloading videos") as pbar:
                for future in as_completed(futures):
                    success, video_id = future.result()
                    if success:
                        successful_downloads += 1
                    else:
                        failed_downloads += 1
                    pbar.update(1)
        
        logging.info(f"Download session completed:")
        logging.info(f"Successful downloads: {successful_downloads}")
        logging.info(f"Failed downloads: {failed_downloads}")
        if failed_downloads > 0:
            logging.info(f"- Check {self.failed_downloads} for details on failed downloads")

        # Create a summary file after download
        self.create_download_summary()

    def create_download_summary(self):
        """Create a summary of the download session"""
        try:
            # Count successful downloads
            successful = 0
            with open(self.archive_file, 'r') as f:
                successful = len(f.readlines())
            
            # Count failed downloads
            failed = 0
            with open(self.failed_downloads, 'r') as f:
                failed = len(f.readlines())
            
            # Create summary
            summary = {
                'download_date': datetime.now().isoformat(),
                'successful_downloads': successful,
                'failed_downloads': failed,
                'total_attempted': successful + failed
            }
            
            # Save summary
            summary_file = self.channel_info_dir / f'download_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
                
        except Exception as e:
            logging.error(f"Error creating download summary: {str(e)}")

def setup_argparse():
    parser = argparse.ArgumentParser(description='Download audio from YouTube channels for transcription')
    parser.add_argument('channel_url', help='YouTube channel URL')
    parser.add_argument('--output-dir', default='data',
                       help='Base output directory (default: data)')
    
    # Cookie options group
    cookie_group = parser.add_mutually_exclusive_group()
    cookie_group.add_argument('--cookies-file', type=str,
                          help='Path to cookies file exported from browser')
    cookie_group.add_argument('--browser-cookies', type=str,
                          choices=['brave', 'chrome', 'chromium', 'edge', 'firefox', 'opera', 'safari', 'vivaldi'],
                          help='Browser to extract cookies from')
    
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Maximum number of concurrent downloads (default: 3)')
    parser.add_argument('--rate-limit', type=float, default=2.0,
                       help='Minimum seconds between requests (default: 2.0)')
    return parser

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    start_time = time.time()
    logging.info(f"Starting download process")
    
    downloader = DownloadManager(
        args.channel_url,
        args.output_dir,
        cookies_file=args.cookies_file,
        browser_cookies=args.browser_cookies,
        max_workers=args.max_workers,
        rate_limit=args.rate_limit
    )
    
    try:
        downloader.download_channel()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    finally:
        elapsed_time = time.time() - start_time
        logging.info(f"Process completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
