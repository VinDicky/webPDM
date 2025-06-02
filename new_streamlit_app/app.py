import streamlit as st
import tempfile
import os
from processor import process_video
import time

# Set custom favicon
st.set_page_config(page_title="Vehicle Counting", page_icon="new_streamlit_app/yolov8.png")

# Load and apply external CSS styles
def load_css(file_path):
    with open(file_path) as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("new_streamlit_app/style.css")

st.markdown(
    """
    <div style="background-color: #2980b9; padding: 15px 0; width: 100vw; position: relative; left: 50%; right: 50%; margin-left: -50vw; margin-right: -50vw; text-align: center; border-radius: 0;">
        <h1 style="color: white; margin: 0; font-family: 'Arial Black', Gadget, sans-serif;">Vehicle Counting with YOLOv8 and Centroid Tracking</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file with .mp4 extension
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_video_path = tfile.name

    st.video(input_video_path, format="video/mp4", start_time=0)

    # Prepare output path
    output_video_path = input_video_path.replace(".mp4", "_output.mp4")

    try:
        with st.spinner("Processing video... This may take a while."):
            counts = process_video(input_video_path, output_video_path)
        st.success("Processing complete!")
    except Exception as e:
        st.error(f"Error during video processing: {e}")
        counts = None

    if counts:
        st.subheader("Vehicle Counts:")
        for label, count in counts.items():
            st.write(f"{label}: {count}")

    # Wait for output video file to be created
    timeout = 30  # seconds
    start_time = time.time()
    while not os.path.exists(output_video_path):
        if time.time() - start_time > timeout:
            st.error("Processed video file not found after waiting.")
            break
        time.sleep(1)
    else:
        # Display processed video
        processed_video_file = open(output_video_path, "rb")
        video_bytes = processed_video_file.read()
        st.video(video_bytes)

        # Download button for processed video
        st.download_button(
            label="Download Processed Video",
            data=video_bytes,
            file_name=os.path.basename(output_video_path),
            mime="video/mp4"
        )

    # Clean up temporary files on app exit
    def cleanup():
        try:
            os.remove(input_video_path)
            os.remove(output_video_path)
        except Exception:
            pass

    st.button("Clear files and reset", on_click=cleanup)
