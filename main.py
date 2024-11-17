import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load the trained YOLO model
model = YOLO('last.pt')

# Streamlit app title
st.title("Helmet Detection")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image using PIL and convert to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # Convert PIL image to NumPy array (OpenCV compatible)
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    
    # Perform inference on the image
    results = model(img_bgr)
    
    # Check for detections with the "helmet" class
    helmet_detected = False
    confidence = None
    
    for detection in results[0].boxes.data:  # Access detections
        class_id = int(detection[-1])  # Class ID is typically the last value
        if class_id == 0:  # Assuming '0' is the class ID for "helmet" in your dataset
            helmet_detected = True
            confidence = detection[-2].item()  # Confidence score is the second-to-last value
            break

    # Display the result
    if helmet_detected:
        st.success(f"Helmet detected with {confidence * 100:.2f}% confidence!")
        # Convert results to an image with bounding boxes rendered
        rendered_img = results[0].plot()
        # Convert BGR back to RGB for displaying in Streamlit
        rendered_img_rgb = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
        st.image(rendered_img_rgb, caption="Detection Results", use_column_width=True)
    else:
        st.error("No helmet detected in the image.")
