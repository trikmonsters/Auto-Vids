import json
import os
import pathlib
import time

from google import genai
from google.genai import types

# =====================================================
# CONFIG
# =====================================================

MODEL = "gemini-3.5-flash"

API_KEY = os.environ["GEMINI_API_KEY"]

AUDIO_FILE = "output/audio/audio.wav"

CHARACTER = os.environ["CHARACTER_REFERENCE"]

STYLE = os.environ["IMAGE_STYLE"]

QUALITY = os.environ["QUALITY"]

OUTPUT_JSON = "output/json/storyboard.json"

# =====================================================
# GEMINI
# =====================================================

client = genai.Client(api_key=API_KEY)

# =====================================================
# UPLOAD AUDIO
# =====================================================

print("Uploading audio...")

audio = client.files.upload(
    file=AUDIO_FILE
)

print(audio.uri)

# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = f"""
You are an Oscar-winning Film Director,
Storyboard Artist,
Concept Artist,
Animation Director,
Visual Psychologist,
Prompt Engineer.

Analyze this AUDIO.

Never summarize.

Think like Pixar.

Think like Dreamworks.

Think like Studio Ghibli.

Think like Disney.

Your job is to convert spoken narration into visual storytelling.

IMPORTANT

DO NOT LIMIT SCENES.

You decide how many scenes are required.

One paragraph may become:

3 scenes

8 scenes

20 scenes

40 scenes

There is NO LIMIT.

Whenever you detect:

- emotional change
- visual object
- action
- movement
- new idea
- metaphor
- comparison
- location change
- subject change
- atmosphere change
- dramatic pause
- cinematic opportunity

Create NEW scene.

Each scene must include:

scene_number

start_timestamp

end_timestamp

duration

importance

emotion

camera

camera_angle

lens

composition

lighting

color_palette

environment

character_action

facial_expression

transition

animation_hint

voice_intensity

visual_focus

2D image prompt

The prompt MUST be production ready.

Use this character reference in EVERY scene:

{CHARACTER}

Art Style:

{STYLE}

Quality:

{QUALITY}

Return ONLY JSON.

Schema:

{{
 "title":"",
 "total_duration":"",
 "scene_count":0,
 "scenes":[]
}}

"""

# =====================================================
# GENERATE
# =====================================================

print("Analyzing audio...")

response = client.models.generate_content(

    model=MODEL,

    contents=[
        SYSTEM_PROMPT,
        audio
    ],

    config=types.GenerateContentConfig(

        temperature=0.35,

        top_p=0.9,

        top_k=40,

        response_mime_type="application/json",

        max_output_tokens=65535
    )
)

text = response.text

# =====================================================
# SAVE JSON
# =====================================================

pathlib.Path("output/json").mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf8"
) as f:

    f.write(text)

print()

print("Storyboard saved.")

print(OUTPUT_JSON)
