import ffmpeg
import os
import time
import whisperx
import torch

# Hardcoded paths
VIDEO_PATH = "/Users/sarjhana/Projects/VBoard-video_transcription/video-transcription-sample.mp4"
OUTPUT_DIR = "transcripts"
AUDIO_PATH = os.path.join(OUTPUT_DIR, "temp_audio.wav")

# Hugging Face token for speaker diarization model
HF_TOKEN = os.environ["HF_TOKEN"]

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Extract audio from video
def extract_audio(video_path, audio_path):
    try:
        ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run(quiet=True, overwrite_output=True)
        print(f"Extracted audio to {audio_path}")
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

# 2. Transcribe + align + diarize with WhisperX
def transcribe_and_diarize(audio_path, hf_token):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and transcribe
    model = whisperx.load_model("base", device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    # Align words
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # Diarize
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)

    # Assign speakers to each word
    result_with_speakers = whisperx.assign_word_speakers(diarize_segments, result_aligned["word_segments"])

    return result_with_speakers

# 3. Save speaker-tagged transcript
def save_transcript(words, output_dir, video_path):
    transcript_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_transcript.txt")
    current_speaker = None
    with open(transcript_path, "w") as f:
        for word in words:
            if word.get("speaker") != current_speaker:
                if current_speaker is not None:
                    f.write("\n")
                current_speaker = word.get("speaker", "Unknown")
                f.write(f"\n[{current_speaker}]: ")
            f.write(word["word"] + " ")
    print(f"Transcript saved to: {transcript_path}")

# Main pipeline
if __name__ == "__main__":
    start_time = time.time()

    print("Extracting audio from video")
    extract_audio(VIDEO_PATH, AUDIO_PATH)

    print("Transcribing, aligning, and diarizing using WhisperX")
    words_with_speakers = transcribe_and_diarize(AUDIO_PATH, HF_TOKEN)

    print("Saving speaker-tagged transcript")
    save_transcript(words_with_speakers, OUTPUT_DIR, VIDEO_PATH)

    end_time = time.time()
    print("Done")
    print(f"Total pipeline execution time: {end_time - start_time:.2f} seconds")
