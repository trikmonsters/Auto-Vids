#!/usr/bin/env python3
"""
analyze_audio.py

Mengirim file audio ke OpenRouter API (model NVIDIA Nemotron 3 Nano Omni)
untuk dianalisis, lalu menghasilkan:
  - Timestamp tiap scene (start_time / end_time)
  - Ringkasan narasi tiap scene
  - Image prompt (untuk AI image generator) tiap scene

Jumlah scene TIDAK dibatasi -> model sendiri yang menentukan berapa banyak
scene berdasarkan isi audio.

Model  : nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
Docs   : https://openrouter.ai/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
         https://openrouter.ai/docs/guides/overview/multimodal/audio
"""

import argparse
import base64
import json
import os
import re
import sys
import time

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

# Ekstensi -> format yang dikenali OpenRouter input_audio.format
SUPPORTED_FORMATS = {
    "wav": "wav",
    "mp3": "mp3",
    "m4a": "m4a",
    "ogg": "ogg",
    "flac": "flac",
    "webm": "webm",
}

SYSTEM_PROMPT = """You are an expert AI film director and storyboard artist.
You will be given a full audio narration (speech / voiceover / podcast / story).

Your task:
1. Listen to the ENTIRE audio from start to finish.
2. Break it into as many narrative "scenes" as the CONTENT naturally requires.
   - Do NOT use a fixed number of scenes.
   - A new scene starts whenever the topic, mood, location, action, or visual
     focus of the narration meaningfully changes.
   - Short audio may need only 2-3 scenes; long audio may need dozens. You
     decide the number based purely on the content, with no upper limit.
3. For every scene, determine the precise start and end timestamp (format
   mm:ss, or hh:mm:ss if the audio is longer than 1 hour), based on what is
   actually being said at that moment in the audio.
4. For every scene, write a single, vivid, highly descriptive English image
   prompt suitable for an AI image generator (subject, action, setting,
   lighting, mood, camera angle, art style) that visually represents what is
   being narrated in that scene.
5. Include a short transcript / summary of the narration spoken during that
   scene.

Respond with ONLY raw JSON, no markdown code fences, no commentary, matching
EXACTLY this schema:

{
  "audio_duration": "mm:ss",
  "total_scenes": <integer>,
  "scenes": [
    {
      "scene": <integer starting at 1>,
      "start_time": "mm:ss",
      "end_time": "mm:ss",
      "narration_text": "<summary / transcript of narration in this segment>",
      "image_prompt": "<detailed visual prompt for an image generator>"
    }
  ]
}

Do not limit the number of scenes. Let the natural structure of the
narration decide how many scenes there are."""

USER_PROMPT = (
    "Analyze the attached audio narration and produce the full scene "
    "breakdown with timestamps and image prompts, following the JSON "
    "schema exactly. Decide the number of scenes yourself based on the "
    "content — do not stop at an arbitrary count."
)


def detect_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in SUPPORTED_FORMATS:
        return SUPPORTED_FORMATS[ext]
    print(
        f"[warn] Ekstensi audio '.{ext}' tidak ada di daftar resmi "
        f"({', '.join(SUPPORTED_FORMATS)}). Mencoba kirim apa adanya, "
        "tapi provider mungkin menolaknya.",
        file=sys.stderr,
    )
    return ext


def encode_audio_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_payload(audio_b64: str, audio_format: str, reasoning: bool, max_tokens: int, temperature: float) -> dict:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": audio_format,
                        },
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning:
        payload["reasoning"] = {"enabled": True}
    return payload


def call_openrouter(api_key: str, payload: dict, max_retries: int = 5) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional, agar muncul di leaderboard OpenRouter (boleh dihapus)
        "HTTP-Referer": "https://github.com",
        "X-Title": "audio-scene-analyzer-github-action",
    }

    backoff = 5
    for attempt in range(1, max_retries + 1):
        resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=600)

        if resp.status_code == 200:
            return resp.json()

        # Rate limit / server sibuk -> retry dengan backoff
        if resp.status_code in (429, 502, 503, 504):
            print(
                f"[retry {attempt}/{max_retries}] HTTP {resp.status_code} dari OpenRouter, "
                f"tunggu {backoff}s...\n{resp.text[:500]}",
                file=sys.stderr,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

        # Error lain -> langsung gagal
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")

    raise RuntimeError("Gagal memanggil OpenRouter API setelah beberapa kali retry.")


def extract_json(text: str) -> dict:
    """Model diminta hanya keluarkan JSON, tapi tetap jaga-jaga kalau ada
    pembungkus ```json ... ``` atau teks tambahan di luar objek JSON."""
    text = text.strip()

    # Coba parse langsung dulu
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Buang code fence kalau ada
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Ambil substring dari '{' pertama sampai '}' terakhir
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Tidak bisa mengekstrak JSON valid dari respons model.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analisis audio -> timestamp + image prompt per scene via OpenRouter")
    parser.add_argument("--input", required=True, help="Path file audio (wav/mp3/m4a/ogg/flac)")
    parser.add_argument("--output", default="output/scenes.json", help="Path file JSON hasil")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"), help="OpenRouter API key")
    parser.add_argument("--reasoning", default="true", help="true/false, aktifkan extended reasoning")
    parser.add_argument("--max-tokens", default="8000", help="Maksimum output token")
    parser.add_argument("--temperature", default="0.4", help="Sampling temperature")
    args = parser.parse_args()

    if not args.api_key:
        print("[error] OPENROUTER_API_KEY tidak ditemukan (set via --api-key atau env var).", file=sys.stderr)
        return 1

    if not os.path.isfile(args.input):
        print(f"[error] File audio tidak ditemukan: {args.input}", file=sys.stderr)
        return 1

    size_mb = os.path.getsize(args.input) / (1024 * 1024)
    print(f"[info] File audio: {args.input} ({size_mb:.2f} MB)")
    if size_mb > 25:
        print(
            "[warn] File audio cukup besar setelah di-encode base64, payload request "
            "bisa sangat besar dan berisiko timeout / ditolak provider. "
            "Pertimbangkan memotong audio jadi beberapa bagian.",
            file=sys.stderr,
        )

    audio_format = detect_format(args.input)
    print(f"[info] Format terdeteksi: {audio_format}")

    audio_b64 = encode_audio_b64(args.input)

    reasoning_enabled = args.reasoning.strip().lower() in ("1", "true", "yes")
    payload = build_payload(
        audio_b64=audio_b64,
        audio_format=audio_format,
        reasoning=reasoning_enabled,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
    )

    print(f"[info] Mengirim request ke OpenRouter ({MODEL_ID})...")
    result = call_openrouter(args.api_key, payload)

    try:
        message_content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"[error] Format respons tidak sesuai harapan: {result}", file=sys.stderr)
        return 1

    try:
        scenes_data = extract_json(message_content)
    except ValueError as e:
        # Simpan raw output supaya tidak hilang, lalu gagal dengan jelas
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        raw_path = args.output + ".raw.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(message_content)
        print(f"[error] {e} Raw output disimpan di {raw_path}", file=sys.stderr)
        return 1

    total_scenes = scenes_data.get("total_scenes", len(scenes_data.get("scenes", [])))
    print(f"[info] Model menghasilkan {total_scenes} scene.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scenes_data, f, ensure_ascii=False, indent=2)

    print(f"[done] Hasil tersimpan di {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
