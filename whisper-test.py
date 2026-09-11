import os
import whisper
import ffmpeg

# === Hardcoded Config ===
MP4_PATH = "/Users/sarjhana/Projects/VBoard-video_transcription/video-transcription-sample.mp4" 
OUTPUT_DIR = "transcripts"

def transcribe_video(mp4_path, output_dir="transcripts"):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract audio
    audio_path = os.path.join(output_dir, "temp_audio.wav")
    try:
        ffmpeg.input(mp4_path).output(audio_path, ac=1, ar='16000').run(quiet=True, overwrite_output=True)
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return

    # Step 2: Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")

    # Step 3: Transcribe
    print("Transcribing...")
    result = model.transcribe(audio_path)

    # Step 4: Save result
    transcript_path = os.path.join(output_dir, os.path.splitext(os.path.basename(mp4_path))[0] + "_transcript.txt")
    with open(transcript_path, "w") as f:
        f.write(result["text"])

    print(f"Transcript saved to: {transcript_path}")

# === Run the function ===
transcribe_video(MP4_PATH, OUTPUT_DIR)
