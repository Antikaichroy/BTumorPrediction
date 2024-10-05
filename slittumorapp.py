import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from collections import Counter

# Load your Keras model
MODEL = keras.models.load_model(r'tumorv4.keras')  # Use raw string to handle backslashes

# Define your class labels (update these according to your model)
CLASS = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']  # Replace with your actual class labels

def predict(files, camera_image=None):
    predictions = []

    # Handle uploaded files
    if files:
        if not (5 <= len(files) <= 10):
            return {"error": "Please upload between 5 and 10 images."}

        for file in files:
            # Read the file as bytes and convert to NumPy array
            file_bytes = file.read()  # Synchronous file read
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Error loading image.")
                continue

            # Preprocess the image as needed
            image = cv2.resize(image, (128, 128))  # Resize according to your model's input shape
            image_batch = np.expand_dims(image, axis=0)
            pred = MODEL.predict(image_batch)
            predicted_class = CLASS[np.argmax(pred[0])]
            predictions.append(predicted_class)

    # Handle camera input
    if camera_image is not None:
        image = np.array(camera_image)
        image = cv2.resize(image, (128, 128))  # Resize according to your model's input shape
        image_batch = np.expand_dims(image, axis=0)
        pred = MODEL.predict(image_batch)
        predicted_class = CLASS[np.argmax(pred[0])]
        predictions.append(predicted_class)

    # Perform majority voting
    if predictions:
        majority_class = Counter(predictions).most_common(1)[0][0]
        confidence = (predictions.count(majority_class) / len(predictions)) * 100
        
        # Display results in separate lines
        return majority_class, confidence
    else:
        return None, "No valid images for prediction."

# Streamlit interface
st.title("Tumor Classification with Deep Learning Model")

# Upload files
uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=['jpg', 'png'])

# Camera input
camera_image = st.camera_input("Capture an image")

# Button to trigger prediction
if st.button("Predict"):
    predicted_class, confidence = predict(uploaded_files, camera_image)
    
    # Check if prediction was successful
    if predicted_class:
        st.write(f"**Class:** {predicted_class}")          # Display class
        st.write(f"**Confidence:** {confidence:.2f}%")    # Display confidence with 2 decimal places
    else:
        st.error(confidence)  # Display error message
