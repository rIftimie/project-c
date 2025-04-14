import os
import subprocess
import logging
from pathlib import Path

def trim_silence(input_file: Path, output_dir: Path = None) -> Path:
    """
    Process an audio file to remove silence using ffmpeg.
    
    Args:
        input_file (Path): Path to input WAV file
        output_dir (Path): Optional output directory. If None, uses input file's directory
        
    Returns:
        Path: Path to the processed file
    """
    try:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # If no output directory specified, use input file's directory
        if output_dir is None:
            output_dir = input_file.parent
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output filename with _trimmed suffix
        output_file = output_dir / f"{input_file.stem}_trimmed{input_file.suffix}"
        
        # ffmpeg command to detect and remove silence
        # Parameters explanation:
        # -af silenceremove: Audio filter to remove silence
        # stop_periods=-1: Process all silence periods
        # stop_duration=1: Minimum silence duration (in seconds) to trigger trimming
        # stop_threshold=-50dB: Sound level threshold below which is considered silence
        command = [
            "ffmpeg",
            "-i", str(input_file),
            "-af", "silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-50dB",
            "-acodec", "pcm_s16le",  # Ensure we maintain WAV format
            "-y",  # Overwrite output file if exists
            str(output_file)
        ]
        
        # Run ffmpeg command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
        # Verify the output file was created
        if not output_file.exists():
            raise RuntimeError("Output file was not created")
            
        # Delete the original file and rename the trimmed version to the original name
        input_file.unlink()
        output_file.rename(input_file)
        
        return input_file
        
    except Exception as e:
        logging.error(f"Error processing audio file {input_file}: {str(e)}")
        raise

def process_audio_directory(directory: Path) -> None:
    """
    Process all WAV files in a directory.
    
    Args:
        directory (Path): Directory containing WAV files to process
    """
    try:
        wav_files = list(directory.glob("*.wav"))
        for wav_file in wav_files:
            try:
                trim_silence(wav_file)
                logging.info(f"Successfully processed {wav_file}")
            except Exception as e:
                logging.error(f"Failed to process {wav_file}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error processing directory {directory}: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio files to remove silence")
    parser.add_argument("input", type=str, help="Input file or directory")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if input_path.is_file():
        trim_silence(input_path)
    elif input_path.is_dir():
        process_audio_directory(input_path)
    else:
        print(f"Invalid input path: {input_path}") 