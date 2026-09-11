import os
import ffmpeg
from faster_whisper import WhisperModel

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

    # Step 2: Load faster-whisper model
    print("Loading faster-whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")  # Change device to "cuda" if you have a GPU

    # Step 3: Transcribe with timestamps
    print("Transcribing...")
    segments, info = model.transcribe(audio_path, beam_size=5)

    # Step 4: Save full transcript and segments with timestamps
    transcript_path = os.path.join(output_dir, os.path.splitext(os.path.basename(mp4_path))[0] + "_transcript.txt")
    segments_path = os.path.join(output_dir, os.path.splitext(os.path.basename(mp4_path))[0] + "_segments.txt")

    full_text = ""
    with open(segments_path, "w") as seg_file:
        for segment in segments:
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            full_text += text + " "
            seg_file.write(f"[{start:.2f} - {end:.2f}] {text}\n")

    with open(transcript_path, "w") as f:
        f.write(full_text.strip())

    print(f"Full transcript saved to: {transcript_path}")
    print(f"Segment timestamps saved to: {segments_path}")

# === Run the function ===
transcribe_video(MP4_PATH, OUTPUT_DIR)
