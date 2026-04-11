import os
import json
import base64
import argparse
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


API_URL = "https://api.openai.com/v1/responses"
MODEL_NAME = "gpt-4.1"


def to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix)
    if not mime_type:
        raise ValueError(f"Unsupported image type: {suffix}")

    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def extract_visible_text_from_html(html_path: str, max_chars: int = 4000) -> str:
    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    return text[:max_chars]


def build_prompt(page_title: str, final_url: str, visible_text: str) -> list[dict]:
    instruction = """
You are a phishing webpage analyst.

Analyze the provided webpage screenshot and page metadata.

Return ONLY valid JSON with exactly this schema:
{
  "brand_guess": "string",
  "task_guess": "string",
  "is_likely_legit": true,
  "is_likely_phishing": false,
  "reasons": ["string", "string"],
  "ui_observations": ["string", "string"],
  "feature_hints": {
    "has_login_form": true,
    "brand_domain_mismatch": false,
    "fake_link_likelihood": "low",
    "mobile_layout_quality": "good"
  }
}

Rules:
- Do not include markdown fences
- Do not include explanation outside JSON
- "task_guess" should be one of: login, checkout, password reset, account verification, product shopping, document share, informational, unknown
- If the page is not performing any user action, use "informational"
- Use visual cues, URL cues, and page text cues
- If uncertain, make the best reasonable guess
"""

    context = f"""Page title: {page_title}
Final URL: {final_url}
Visible text excerpt:
{visible_text}
"""

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": "You are a security analyst specialized in phishing-page identification. Return only strict JSON."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction},
                {"type": "input_text", "text": context},
            ],
        },
    ]


def extract_output_text(response_json: dict) -> Optional[str]:
    try:
        output_items = response_json.get("output", [])
        collected = []

        for item in output_items:
            if item.get("type") != "message":
                continue

            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    collected.append(content.get("text", ""))

        if collected:
            return "\n".join(collected).strip()

    except Exception:
        return None

    return None


def try_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract JSON object from noisy text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def analyze(
    image_path: str,
    page_title: str,
    final_url: str,
    visible_text: str,
    timeout: int = 60,
) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt_messages = build_prompt(
        page_title=page_title,
        final_url=final_url,
        visible_text=visible_text[:4000],
    )

    # Add image into the user message
    prompt_messages[1]["content"].append(
        {
            "type": "input_image",
            "image_url": to_data_url(image_path),
        }
    )

    payload = {
        "model": MODEL_NAME,
        "input": prompt_messages,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)

    if not response.ok:
        print("Status:", response.status_code)
        print("Response:", response.text)
        response.raise_for_status()

    response_json = response.json()
    output_text = extract_output_text(response_json)
    parsed_json = try_parse_json(output_text or "")

    return {
        "raw_response": response_json,
        "model_output_text": output_text,
        "parsed_result": parsed_json,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze a captured webpage with OpenAI.")
    parser.add_argument("--image", required=True, help="Path to screenshot image")
    parser.add_argument("--title", required=True, help="Page title")
    parser.add_argument("--url", required=True, help="Final URL")
    parser.add_argument("--text", help="Visible text directly")
    parser.add_argument("--html", help="Path to HTML file to extract visible text from")
    parser.add_argument("--save", help="Optional output JSON file path")
    args = parser.parse_args()

    visible_text = args.text
    if not visible_text and args.html:
        visible_text = extract_visible_text_from_html(args.html)
    if not visible_text:
        visible_text = ""

    result = analyze(
        image_path=args.image,
        page_title=args.title,
        final_url=args.url,
        visible_text=visible_text,
    )

    print("\n=== MODEL OUTPUT TEXT ===")
    print(result["model_output_text"])

    print("\n=== PARSED RESULT ===")
    print(json.dumps(result["parsed_result"], indent=2))

    if args.save:
        out_path = Path(args.save)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved full result to: {out_path}")


if __name__ == "__main__":
    main()