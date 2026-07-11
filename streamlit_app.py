import streamlit as st
import tempfile
import os
import video_utils
import ai_inference

st.set_page_config(page_title="Quiptionary", page_icon="🎬")
st.title("Quiptionary")
st.caption("it doesn't caption your video — it performs it")

video_url = st.text_input("Video URL", placeholder="https://example.com/clip.mp4")
styles = st.multiselect(
    "Styles to generate",
    ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"],
    default=["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
)

if st.button("Generate captions", type="primary"):
    if not video_url:
        st.error("Paste a video URL first.")
    elif not styles:
        st.error("Pick at least one style.")
    else:
        temp_video_path = tempfile.mktemp(suffix=".mp4")
        with st.spinner("Downloading video..."):
            success = video_utils.download_video(video_url, temp_video_path)

        if not success:
            st.error("Couldn't download that video. Check the URL.")
        else:
            with st.spinner("Extracting frames..."):
                frame_paths = video_utils.extract_frames(temp_video_path, max_frames=10)

            if not frame_paths:
                st.error("Couldn't extract frames from this video.")
            else:
                for style in styles:
                    with st.spinner(f"Generating '{style}' caption..."):
                        caption = ai_inference.generate_caption(frame_paths, style)
                    st.subheader(style.replace("_", " "))
                    if caption.startswith("[CAPTION_FAILED"):
                        st.warning(caption)
                    else:
                        st.write(caption)

                for path in frame_paths:
                    if os.path.exists(path):
                        os.remove(path)
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)