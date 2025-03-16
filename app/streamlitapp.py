import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf

from utils import load_data, num_to_char
from modelutil import create_lipnet_model

# Set Streamlit layout
st.set_page_config(layout='wide')

# Sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is based on the LipNet deep learning model.')

st.title('LipNet Full Stack App')

# Ensure the video dataset directory exists
video_dir = os.path.join("..", "data", "s1")
if not os.path.exists(video_dir):
    st.error(f"⚠️ Video dataset directory not found: {video_dir}")
    st.stop()

# List available videos
options = os.listdir(video_dir)
selected_video = st.selectbox('Choose video', options) if options else None

# Display columns
col1, col2 = st.columns(2)

if selected_video:
    file_path = os.path.join(video_dir, selected_video)

    with col1:
        st.info('🎥 The video below displays the converted video in MP4 format')

        # Convert to mp4 format if needed
        converted_video_path = "test_video.mp4"
        os.system(f'ffmpeg -i "{file_path}" -vcodec libx264 "{converted_video_path}" -y')

        # Show video in Streamlit
        with open(converted_video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)

    with col2:
        st.info("🧠 This is all the model sees when making a prediction")

        try:
            # Load and preprocess video
            video_frames, annotations = load_data(tf.convert_to_tensor(file_path))

            # Debugging info
            st.text(f"📏 Video Frames Shape: {video_frames.shape}")
            st.text(f"📝 Annotations Type: {type(annotations)}")

            # Convert frames to GIF format for visualization
            processed_frames = [(frame.numpy() * 255).astype(np.uint8).squeeze() for frame in video_frames]
            gif_path = "animation.gif"
            imageio.mimsave(gif_path, processed_frames, fps=10)
            st.image(gif_path, width=400)

        except Exception as e:
            st.error(f"⚠️ Error loading video data: {e}")
            st.stop()

    st.info("🔍 This is the output of the machine learning model as tokens")

    try:
        # Load the model
        model = create_lipnet_model()

        # Perform prediction
        yhat = model.predict(tf.expand_dims(video_frames, axis=0))
        decoded = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

        # Convert prediction to text
        predicted_text = tf.strings.reduce_join([num_to_char(x) for x in decoded]).numpy().decode('utf-8')

        st.text("📝 Prediction:")
        st.text(predicted_text)

    except Exception as e:
        st.error(f"⚠️ Error during model prediction: {e}")
