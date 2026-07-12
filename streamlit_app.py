import streamlit as st
import tempfile
import os
import video_utils
import ai_inference

st.set_page_config(page_title="Quiptionary", page_icon="🎬", layout="centered")
st.title("Quiptionary")
st.caption("it doesn't caption your video — it performs it")

# Same palette as the cover image and slide deck
STYLE_COLORS = {
    "formal": "#3554D1",
    "sarcastic": "#8B3FD1",
    "humorous_tech": "#D18A2E",
    "humorous_non_tech": "#D1522E",
}

def render_caption_box(style, caption):
    color = STYLE_COLORS.get(style, "#555555")
    label = style.replace("_", " ")
    if caption.startswith("[CAPTION_FAILED"):
        st.warning(f"**{label}**\n\n{caption}")
        return
    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {color};
            background-color: rgba(255,255,255,0.04);
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 14px;
        ">
            <div style="font-weight:700; font-size:15px; color:{color}; margin-bottom:6px;">
                {label}
            </div>
            <div style="font-size:14.5px; line-height:1.5;">
                {caption}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Input: URL or file upload ---
tab_url, tab_upload = st.tabs(["Paste a URL", "Upload a video"])

video_url = None
uploaded_file = None

with tab_url:
    video_url = st.text_input("Video URL", placeholder="https://example.com/clip.mp4")

with tab_upload:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "webm", "mkv"])

styles = st.multiselect(
    "Styles to generate",
    ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
    default=["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
)

generate = st.button("Generate captions", type="primary")

if generate:
    if not styles:
        st.error("Pick at least one style.")
    elif not video_url and not uploaded_file:
        st.error("Paste a video URL or upload a file first.")
    else:
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        progress = st.progress(0, text="Getting video ready...")

        # --- Step 1: get the video onto disk ---
        if uploaded_file is not None:
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.read())
            got_video = True
        else:
            got_video = video_utils.download_video(video_url, temp_video_path)

        if not got_video:
            progress.empty()
            st.error("Couldn't load that video. Check the URL and try again.")
        else:
            # --- Step 2: extract frames ---
            progress.progress(15, text="Extracting frames from the clip...")
            frame_paths = video_utils.extract_frames(temp_video_path, max_frames=10)

            if not frame_paths:
                progress.empty()
                st.error("Couldn't extract frames from this video.")
            else:
                progress.progress(30, text=f"Got {len(frame_paths)} frames. Starting captions...")

                # --- Step 3: generate each style, advancing the bar per style ---
                results = {}
                step = 65 // max(len(styles), 1)
                current = 30

                for style in styles:
                    label = style.replace("_", " ")
                    progress.progress(current, text=f"Writing the '{label}' caption...")
                    caption = ai_inference.generate_caption(frame_paths, style)
                    results[style] = caption
                    current = min(current + step, 99)

                progress.progress(100, text="Done!")
                progress.empty()

                st.success("Captions ready!")

                # --- Captions first, in colored boxes ---
                for style in styles:
                    render_caption_box(style, results[style])

                # --- Video preview shown after generation, alongside the results ---
                st.divider()
                st.subheader("Source clip")
                st.video(temp_video_path)

                for path in frame_paths:
                    if os.path.exists(path):
                        os.remove(path)

            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)