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
    # Using strict Markdown headings forces reasoning models to obey
    guard = (
        "\n\n### CRITICAL INSTRUCTIONS ###\n"
        "1. You MUST wrap your final caption inside exact <caption_output> and </caption_output> tags.\n"
        "2. Do NOT put anything else inside the tags.\n"
        "3. Do NOT explain your thinking or write a checklist."
    )
    
    prompts = {
        "formal": "Analyse the visual, with the cold attitude of HAL-9000 with purely factual, emotionless tone.",
        "sarcastic": "Analyze the visual with a very deadpan and sarcastic tone and eye-rolling, as if you were forced to deal with mere mortals with a sigh incomparable to your power, describe with incredible wit and condescending tone.",
        "humorous_tech": "Describe the visual like a tired millenial of workload in his AI job. But focus on the visual, do not divert from the visual heavily towards your 'job'. Use clever technical jargon but make it effective in being absolutely funny and understandably humorous.",
        "humorous_non_tech": "Give a very funny and relatable attitude when describing the visual sequence as if you were a man who is in his 50s who finds it hard to keep up with the fast growing world. Do not use technical jargon and do not give very niche references."
    }
    
    # This securely glues the strict rules to the end of every prompt
    return prompts.get(style, f"Caption this in a {style} tone.") + guard

def sample(frame_paths, num_samples=4):
    '''Gets num_samples evenly spaced from multiple elements of a set, not just a single frame'''
    total=len(frame_paths)
    if total<=num_samples:
        return frame_paths
    
    selected=[]
    for i in range(num_samples): 
        # Spread i evenly across the range [0, total-1]
        # i=0 always lands on index 0 (first frame)
        # i=num_samples-1 always lands on index total-1 (last frame)
        pos=i*(total-1)/(num_samples-1)
        index=round(pos)
        selected.append(frame_paths[index])
    return selected
    
        
    
    

def generate_caption(frame_paths, style):
    """Sends the compressed image and text prompt to a Fireworks Vision model."""
    print(f"Generating '{style}' caption from Qwen....")
    
    try:
        # sampling several frames from the clip, especially for longer ones
        sample_frames = sample(frame_paths, num_samples=4)

        # Construct the actual content multimodal payload
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(frame_path)}"
                }
            }
            for frame_path in sample_frames
        ]
        content.append({
            "type": "text",
            "text": get_style_prompt(style)
        })
        
        # Pinging Qwen Vision model
        response = client.chat.completions.create(
            model="accounts/fireworks/models/qwen3p7-plus",
            messages=[
                {
                    # Forces Qwen to output clean XML tags
                    "role": "system",
                    "content": (
                        "You are a strict data-formatting pipeline. You will receive a persona and an image. "
                        "You MUST wrap your final caption inside exact <caption_output> and </caption_output> tags. "
                        "Do NOT output any thinking process. Do NOT output any conversational text."
                        "Do NOT use Markdown formatting (no asterisks, no headers). Plain text only. Respond ONLY IN ENGLISH"
                    )
                },
                {
                    "role": "user",
                    "content": content
                    
                }
            ],
            max_tokens=400,
            temperature=0.7,
            extra_body={"reasoning_effort": "none"}
        )
        
        raw_output = response.choices[0].message.content.strip()
        
        # Find for the exact tags and extract only the content inside
        match = re.search(r'<caption_output>(.*?)</caption_output>', raw_output, re.DOTALL)
        
        if match:
            # Clean extraction
            caption= match.group(1).strip()
            if caption:
                return caption
        
       
        print(f"  -> [WARNING] AI missed or emptied the XML tags for style '{style}'.")
        print(f"  -> [RAW OUTPUT] {raw_output[:300]}")
        return f"[CAPTION_FAILED: '{style}' - malformed model output, see logs]"
            
    except Exception as e:
        print(f"  -> API Error: {e}")
        return f"[CAPTION FAILED: '{style}' - API ERROR: {e}]"