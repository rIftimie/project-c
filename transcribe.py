import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from faster_whisper import WhisperModel
import torch
from tqdm import tqdm

class TranscriptionManager:
    def __init__(self, model_size: str = "medium", device: str = "cpu", compute_type: str = "int8", 
                 cpu_threads: int = 12, num_workers: int = 4, batch_size: int = 8):
        """
        Initialize the transcription manager.
        
        Args:
            model_size: Size of the Whisper model to use (tiny, base, small, medium, large-v3)
            device: Device to use for inference ("cpu", "cuda", "auto")
            compute_type: Type of compute to use ("float16", "float32", "int8")
            cpu_threads: Number of CPU threads to use for computation
            num_workers: Number of workers for parallel processing
            batch_size: Batch size for processing audio segments
        """
        self.device = "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.compute_type = compute_type if self.device == "cuda" else compute_type
        
        logging.info(f"Loading Whisper model {model_size} on {self.device}")
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )
        self.batch_size = batch_size

    def transcribe_audio(self, audio_path: Path, output_dir: Optional[Path] = None, output_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file and save the results.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save transcription results (defaults to audio file directory)
            output_filename: Custom filename for the output (without extension)
            
        Returns:
            Dict containing transcription results
        """
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Default output directory to audio file location if not specified
            if output_dir is None:
                output_dir = audio_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare transcription results
            transcription_data = {
                "text": "",
                "segments": [],
                "language": None,
                "language_probability": None
            }

            # Perform transcription with progress bar
            logging.info(f"Transcribing {audio_path}")
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size=self.batch_size,  # Use batch_size for beam_size
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Update language info
            transcription_data["language"] = info.language
            transcription_data["language_probability"] = info.language_probability

            # Process segments with progress bar
            # First, collect segments to get total count
            segments_list = []
            print("Collecting segments...")
            for segment in segments:
                segments_list.append(segment)

            # Now process segments with progress bar
            print("Processing segments...")
            for segment in tqdm(segments_list, desc="Processing segments", unit="segment", ncols=100):
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability
                        }
                        for word in segment.words
                    ]
                }
                transcription_data["segments"].append(segment_data)
                transcription_data["text"] += segment.text + " "

            # Save transcription results
            output_name = f"{output_filename if output_filename else audio_path.stem}.json"
            output_file = output_dir / output_name
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(transcription_data, f, indent=2, ensure_ascii=False)

            logging.info(f"Transcription saved to {output_file}")
            return transcription_data

        except Exception as e:
            logging.error(f"Error transcribing {audio_path}: {str(e)}")
            raise

    def process_directory(self, input_dir: Path, output_dir: Optional[Path] = None) -> None:
        """
        Process all WAV files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save transcription results
        """
        try:
            wav_files = list(input_dir.glob("*.wav"))
            total_files = len(wav_files)
            logging.info(f"Found {total_files} WAV files to process")
            
            for wav_file in tqdm(wav_files, desc="Processing files", unit="file", ncols=100):
                try:
                    self.transcribe_audio(wav_file, output_dir)
                except Exception as e:
                    logging.error(f"Failed to process {wav_file}: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Error processing directory {input_dir}: {str(e)}")
            raise

if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('transcription.log'),
            logging.StreamHandler()
        ]
    )
    
    parser = argparse.ArgumentParser(description="Transcribe audio files using faster-whisper")
    parser.add_argument("input", type=str, help="Input audio file or directory")
    parser.add_argument("--output-dir", type=str, help="Output directory for transcriptions")
    parser.add_argument("--model-size", type=str, default="medium",
                      choices=["tiny", "base", "small", "medium", "large-v3"],
                      help="Size of the Whisper model to use")
    parser.add_argument("--device", type=str, default="cpu",
                      choices=["cpu", "cuda", "auto"],
                      help="Device to use for inference")
    parser.add_argument("--compute-type", type=str, default="int8",
                      choices=["float16", "float32", "int8"],
                      help="Type of compute to use")
    parser.add_argument("--cpu-threads", type=int, default=12,
                      help="Number of CPU threads to use for computation")
    parser.add_argument("--num-workers", type=int, default=4,
                      help="Number of workers for parallel processing")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for processing audio segments")
    
    args = parser.parse_args()
    
    # Initialize transcription manager
    transcriber = TranscriptionManager(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if input_path.is_file():
        transcriber.transcribe_audio(input_path, output_dir)
    elif input_path.is_dir():
        transcriber.process_directory(input_path, output_dir)
    else:
        print(f"Invalid input path: {input_path}") 