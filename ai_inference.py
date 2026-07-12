import os
import base64
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ.get("FIREWORKS_API_KEY")
)


def encode_image(image_path):
    """Converts a local image file into a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_style_prompt(style):
    '''
    prompts for different styles
    '''
    # Forces the model to ground the caption in one real, specific visual
    # detail before layering on persona, instead of leaning purely on
    # mood/metaphor and drifting away from what's actually in the video.
    anchor = (
        "\n\nBefore writing, silently pick ONE specific, vivid detail you actually see "
        "in the frames - a color, an object, an action, text/signage, a number, anything concrete. "
        "Build your one-line caption around that specific detail, filtered through your persona's voice. "
        "A caption anchored in one real detail beats one built purely on mood or metaphor."
    )

    guard = (
        "\n\n### CRITICAL INSTRUCTIONS ###\n"
        "1. You MUST wrap your final caption inside exact <caption_output> and </caption_output> tags.\n"
        "2. Do NOT put anything else inside the tags.\n"
        "3. Do NOT explain your thinking or write a checklist.\n"
        "4. Do NOT use Markdown formatting (no asterisks, no headers, no bullet points). Plain text only.\n"
        "5. Go straight into the caption - no preamble like 'Bug Report:' or 'Caption:' before the tag.\n"
        "6. Keep the caption to 1-2 sentences, no more than about 30 words total. One punchy, "
        "specific line beats a paragraph - like a witty caption under a photo, not a description."
    )

    prompts = {
        "formal": "Analyse the visual, with the cold attitude of HAL-9000 with purely factual, emotionless tone.",
        "sarcastic": "Analyze the visual with a very deadpan and sarcastic tone and eye-rolling, as if you were forced to deal with mere mortals with a sigh incomparable to your power, describe with incredible wit and condescending tone.",
        "humorous_tech": "Describe the visual like a tired millenial of workload in his AI job. But focus on the visual, do not divert from the visual heavily towards your 'job'. Use clever technical jargon but make it effective in being absolutely funny and understandably humorous.",
        "humorous_non_tech": "Give a very funny and relatable attitude when describing the visual sequence as if you were a man who is in his 50s who finds it hard to keep up with the fast growing world. Do not use technical jargon and do not give very niche references."
    }

    return prompts.get(style, f"Caption this in a {style} tone.") + anchor + guard


def sample_frames(frame_paths, num_samples=4):
    '''Gets num_samples evenly spaced frames from the full extracted set, not just a single frame'''
    total = len(frame_paths)
    if total <= num_samples:
        return frame_paths

    selected = []
    for i in range(num_samples):
        # Spread i evenly across the range [0, total-1]
        # i=0 always lands on index 0 (first frame)
        # i=num_samples-1 always lands on index total-1 (last frame)
        pos = i * (total - 1) / (num_samples - 1)
        index = round(pos)
        selected.append(frame_paths[index])
    return selected


def generate_caption(frame_paths, style):
    """Sends multiple sampled frames and the text prompt to a Fireworks Vision model."""
    print(f"Generating '{style}' caption from Qwen....")

    try:
        # Sample several frames across the clip instead of just the middle one,
        # so longer clips (up to 2 min) aren't captioned from a single instant
        sample = sample_frames(frame_paths, num_samples=4)

        # Build one image_url block per sampled frame
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(frame_path)}"
                }
            }
            for frame_path in sample
        ]
        # Text prompt goes last, after all the images
        content.append({
            "type": "text",
            "text": get_style_prompt(style)
        })

        response = client.chat.completions.create(
            model="accounts/fireworks/models/qwen3p7-plus",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict data-formatting pipeline. You will receive a persona and images. "
                        "You MUST wrap your final caption inside exact <caption_output> and </caption_output> tags. "
                        "Do NOT output any thinking process. Do NOT output any conversational text. "
                        "Do NOT use Markdown formatting (no asterisks, no headers). Plain text only. "
                        "Respond only in English."
                    )
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=120,
            temperature=0.7,
            extra_body={"reasoning_effort": "none"}
        )

        raw_output = response.choices[0].message.content.strip()

        # Find the exact tags and extract only the content inside
        match = re.search(r'<caption_output>(.*?)</caption_output>', raw_output, re.DOTALL)

        if match:
            caption = match.group(1).strip()
            if caption:
                return caption
            # tags present but empty - falls through to the failure marker below

        print(f"  -> [WARNING] AI missed or emptied the XML tags for style '{style}'.")
        print(f"  -> [RAW OUTPUT] {raw_output[:300]}")
        return f"[CAPTION_FAILED: '{style}' - malformed model output, see logs]"

    except Exception as e:
        print(f"  -> API Error: {e}")
        return f"[CAPTION_FAILED: '{style}' - API error: {e}]"


def generate_description(frame_paths):
    """
    Generates a plain, neutral description of the video - no persona,
    just what's actually happening. Used by the Streamlit demo UI to show
    alongside the styled captions, so viewers can compare each persona
    against the literal content. Not used by main.py / the Docker submission,
    since the judged output schema only requires the 4 styled captions.
    """
    try:
        sample = sample_frames(frame_paths, num_samples=4)

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(frame_path)}"
                }
            }
            for frame_path in sample
        ]
        content.append({
            "type": "text",
            "text": (
                "Describe exactly what is happening in this video sequence in one plain, "
                "neutral sentence. State only concrete visual facts - objects, people, "
                "setting, actions, colors, text/signage. No opinion, no tone, no personality."
                "\n\nWrap your answer in exact <caption_output> and </caption_output> tags, "
                "nothing else inside them. No Markdown. Respond only in English."
            )
        })

        response = client.chat.completions.create(
            model="accounts/fireworks/models/qwen3p7-plus",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict data-formatting pipeline. You will receive images. "
                        "You MUST wrap your final answer inside exact <caption_output> and "
                        "</caption_output> tags. Do NOT output any thinking process."
                    )
                },
                {"role": "user", "content": content}
            ],
            max_tokens=100,
            temperature=0.3,
            extra_body={"reasoning_effort": "none"}
        )

        raw_output = response.choices[0].message.content.strip()
        match = re.search(r'<caption_output>(.*?)</caption_output>', raw_output, re.DOTALL)

        if match:
            description = match.group(1).strip()
            if description:
                return description

        return "[DESCRIPTION_FAILED: malformed model output]"

    except Exception as e:
        print(f"  -> API Error (description): {e}")
        return f"[DESCRIPTION_FAILED: API error: {e}]"