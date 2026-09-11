import ffmpeg
import whisper
from pyannote.audio import Pipeline
import numpy as np
import os
import time

# Hardcoded paths
VIDEO_PATH = "/Users/sarjhana/Projects/VBoard-video_transcription/video-transcription-sample.mp4"
OUTPUT_DIR = "transcripts"
AUDIO_PATH = os.path.join(OUTPUT_DIR, "temp_audio.wav")

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

# 2. Run diarization pipeline
def diarize(audio_path, hf_token):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=hf_token)
    diarization = pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments

# 3. Run Whisper transcription with word timestamps
def transcribe(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    words = []
    for segment in result["segments"]:
        for w in segment["words"]:
            words.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"]
            })
    return words

# 4. Assign speakers to each word by timestamp matching
def assign_speakers(words, diarization_segments):
    for word in words:
        word_start = word["start"]
        speaker = "Unknown"
        for seg in diarization_segments:
            if seg["start"] <= word_start <= seg["end"]:
                speaker = seg["speaker"]
                break
        word["speaker"] = speaker
    return words

# 5. save transcript file with speaker labels
def save_transcript(words, output_dir, video_path):
    transcript_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_transcript.txt")
    current_speaker = None
    with open(transcript_path, "w") as f:
        for word in words:
            if word["speaker"] != current_speaker:
                if current_speaker is not None:
                    f.write("\n")
                current_speaker = word["speaker"]
                f.write(f"\n[{current_speaker}]: ")
            f.write(word["word"] + " ")
    print(f"Transcript saved to: {transcript_path}")


if __name__ == "__main__":
    
    start_time = time.time()
    HF_TOKEN = os.environ["HF_TOKEN"]

    print("Extracting audio from video")
    extract_audio(VIDEO_PATH, AUDIO_PATH)

    print("Diarizing audio using pyannote")
    diarization_segments = diarize(AUDIO_PATH, HF_TOKEN)

    print("Transcribing audio using whisper")
    words = transcribe(AUDIO_PATH)

    print("Assigning speakers to words")
    words_with_speakers = assign_speakers(words, diarization_segments)

    print("Saving speaker-tagged transcript") 
    save_transcript(words_with_speakers, OUTPUT_DIR, VIDEO_PATH)
    end_time = time.time()

    print("Done")
    elapsed = end_time - start_time
    print(f"Total pipeline execution time: {elapsed:.2f} seconds")
