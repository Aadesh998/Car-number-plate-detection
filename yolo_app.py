import pandas as pd
import cv2
from torch import device
from ultralytics import YOLO
import streamlit as st
import numpy as np
import os

st.title("Car licence plate detection in Image and Video")

# Upload image or video file
uploaded_file = st.file_uploader("Upload Image and video file", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Load the YOLO model for license plate detection
try:
    model = YOLO(r'C:\Users\Dell\Desktop\Licence plate Detection\best.pt')
except Exception as e:
    st.markdown(f"An error occurred while loading the model {e}")

# Function to predict and annotate video frames, then save the output video
def predict_and_plot_video(video_file, output_path):
    """
    Process a video file for license plate detection, annotate detected plates,
    and save the output video.
    
    Args:
        video_file (str): Path to the input video file.
        output_path (str): Path to save the output annotated video.

    Returns:
        str: Path to the saved output video.
    """
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            st.markdown("Error opening video file: {video_file}")
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(rgb_frame, device='cpu')
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            out.write(frame)
        cap.release()
        out.release()
        return output_path
        
    except Exception as e:
        st.markdown("Error processing video: {e}")

# Function to predict and annotate image, then save the output image
def predict_and_save_image(input_file, output_file):
    """
    Process an image file for license plate detection, annotate detected plates,
    and save the output image.
    
    Args:
        input_file (str): Path to the input image file.
        output_file (str): Path to save the output annotated image.

    Returns:
        str: Path to the saved output image.
    """
    try:
        results = model.predict(input_file, device='cpu')
        image = cv2.imread(input_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_file, image)
        return output_file
            
    except Exception as e:
        st.markdown(f'Error: {e}')
        return None

# Function to process the uploaded media file (image or video)
def process_media(input_file_path, output_file_path):
    """
    Determine the file type of the uploaded media and process accordingly.
    
    Args:
        input_file_path (str): Path to the input media file.
        output_file_path (str): Path to save the processed output file.

    Returns:
        str: Path to the saved output file.
    """
    file_extension = os.path.splitext(input_file_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_file_path, output_file_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_file_path, output_file_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

# Handle the uploaded file and process it
if uploaded_file is not None:
    input_file = os.path.join("temp", uploaded_file.name)
    output_file = os.path.join('temp', f'output_{uploaded_file.name}')
    try:
        with open(input_file, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.write("Processing.....")
        result_path = process_media(input_file, output_file)
        if result_path is not None:
            if result_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_file = open(result_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                st.image(result_path)
    except Exception as e:
        st.markdown(f'An error occurred in result path: {e}')
