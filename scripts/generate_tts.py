import os
import re
import time
import wave
import struct

from pathlib import Path

from google import genai
from google.genai import types

# ============================================================
# CONFIG
# ============================================================

MODEL = "gemini-3.1-flash-tts-preview"

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2

VOICE = os.environ["VOICE_NAME"]
LANGUAGE = os.environ["LANGUAGE"]

INPUT_FILE = "narration.txt"

OUTPUT_DIR = Path("output/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_AUDIO = OUTPUT_DIR / "audio.wav"

# ============================================================
# GEMINI
# ============================================================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# ============================================================
# LOAD TEXT
# ============================================================

with open(INPUT_FILE, encoding="utf8") as f:
    raw_text = f.read().strip()

paragraphs = [
    p.strip()
    for p in raw_text.split("\n\n")
    if p.strip()
]

print(f"{len(paragraphs)} paragraphs")

# ============================================================
# SPLIT LONG PARAGRAPH
# ============================================================

def split_long_text(text, limit=1800):

    if len(text) < limit:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []

    current = ""

    for s in sentences:

        if len(current) + len(s) < limit:

            current += " " + s

        else:

            chunks.append(current.strip())

            current = s

    if current:

        chunks.append(current.strip())

    return chunks

# ============================================================
# STYLE PROMPT
# ============================================================

def build_prompt(text, index, total):

    if index == 0:

        mood = """
Speak warmly.

Slow.

Confident.

Professional psychologist.

Very natural.

Comfortable pacing.

"""

    elif index == total-1:

        mood = """
End gently.

Hopeful.

Warm.

Emotional.

"""

    else:

        mood = """
Natural narration.

Professional documentary.

Warm.

Relaxed.

"""

    return f"""
Language: {LANGUAGE}

Read EXACTLY the narration.

Do not change words.

Do not summarize.

Maintain punctuation naturally.

{mood}

Narration:

{text}
"""

# ============================================================
# SILENCE
# ============================================================

def silence(seconds):

    samples = int(seconds * SAMPLE_RATE)

    return struct.pack(
        f"<{samples}h",
        *([0]*samples)
    )

# ============================================================
# SAVE WAV
# ============================================================

def save_wave(path, pcm):

    with wave.open(str(path), "wb") as wf:

        wf.setnchannels(CHANNELS)

        wf.setsampwidth(SAMPLE_WIDTH)

        wf.setframerate(SAMPLE_RATE)

        wf.writeframes(pcm)

# ============================================================
# GENERATE
# ============================================================

all_pcm = []

scene = 1

for p_index, para in enumerate(paragraphs):

    chunks = split_long_text(para)

    for chunk in chunks:

        prompt = build_prompt(
            chunk,
            p_index,
            len(paragraphs)
        )

        print(f"Scene {scene}")

        for retry in range(5):

            try:

                response = client.models.generate_content(

                    model=MODEL,

                    contents=prompt,

                    config=types.GenerateContentConfig(

                        response_modalities=["AUDIO"],

                        speech_config=types.SpeechConfig(

                            voice_config=types.VoiceConfig(

                                prebuilt_voice_config=types.PrebuiltVoiceConfig(

                                    voice_name=VOICE

                                )
                            )
                        )
                    )
                )

                pcm = response.candidates[0].content.parts[0].inline_data.data

                all_pcm.append(pcm)

                break

            except Exception as e:

                wait = 2**retry

                print(e)

                print(f"Retry {wait}s")

                time.sleep(wait)

        all_pcm.append(silence(1.2))

        scene += 1

# ============================================================
# EXPORT
# ============================================================

combined = b"".join(all_pcm)

save_wave(FINAL_AUDIO, combined)

duration = len(combined) / SAMPLE_RATE / SAMPLE_WIDTH

print()

print("="*50)

print("DONE")

print(FINAL_AUDIO)

print(f"Duration : {duration:.2f} sec")

print("="*50)

metadata = f"""
Model : {MODEL}
Voice : {VOICE}
Language : {LANGUAGE}
Duration : {duration}
Paragraphs : {len(paragraphs)}
"""

(Path("output")/"metadata.txt").write_text(
    metadata,
    encoding="utf8"
)
