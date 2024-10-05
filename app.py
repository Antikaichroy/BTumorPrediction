from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from collections import Counter
import cv2
import numpy as np
import asyncio



app = FastAPI()

# Load your model (update the path as needed)
MODEL = tf.keras.models.load_model("tumorv4.keras")
CLASS = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Frontend HTML
@app.get("/", response_class=HTMLResponse)
async def get_html():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Prediction</title>
        <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
            }
            .container {
                margin-top: 50px;
            }
            .upload-form {
                border: 2px dashed #007bff;
                padding: 20px;
                border-radius: 5px;
                background-color: white;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .upload-button {
                margin-top: 20px;
            }
            .result {
                margin-top: 30px;
                padding: 15px;
                border: 1px solid #007bff;
                border-radius: 5px;
                background-color: #e9ecef;
            }
            .uploaded-image {
                margin-top: 20px;
                max-width: 100%;
                height: auto;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            .uploaded-images {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 20px;
            }
            .uploaded-images img {
                max-width: 150px;
                height: auto;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="upload-form text-center">
                <h2 class="mb-4">Upload Pictures for Brain Tumor Prediction</h2>
                <input type="file" id="file-input" accept="image/*" class="form-control-file" multiple required>
                <button class="btn btn-primary upload-button" id="upload-button">Upload</button>
                <div class="uploaded-images" id="uploaded-images"></div>
            </div>
            <div class="result" id="result" style="display: none;">
                <h3>Prediction Result:</h3>
                <p id="class"></p>
                <p id="confidence"></p>
            </div>
        </div>

        <script>
            // Display selected images
            document.getElementById('file-input').onchange = function(event) {
                const files = event.target.files;
                const imagesDiv = document.getElementById('uploaded-images');
                imagesDiv.innerHTML = '';  // Clear any previously uploaded images

                Array.from(files).forEach(file => {
                    const img = document.createElement('img');
                    img.src = URL.createObjectURL(file);
                    imagesDiv.appendChild(img);
                });
            };

            // Upload and predict
            document.getElementById('upload-button').onclick = async function() {
                const fileInput = document.getElementById('file-input');
                const resultDiv = document.getElementById('result');
                const classElem = document.getElementById('class');
                const confidenceElem = document.getElementById('confidence');

                // Check if the number of files is between 5 and 10
                if (fileInput.files.length < 5 || fileInput.files.length > 10) {
                    alert('Please upload between 5 and 10 images.');
                    return;  // Prevent further execution if validation fails
                }

                const formData = new FormData();
                Array.from(fileInput.files).forEach(file => {
                    formData.append("files", file);  // Append multiple files
                });

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('Error occurred while uploading the images.');
                    }

                    const data = await response.json();
                    classElem.textContent = `Class: ${data.Class}`;
                    confidenceElem.textContent = `Confidence: ${data.Confidence.toFixed(2)}%`;
                    resultDiv.style.display = 'block';
                } catch (error) {
                    alert(error.message);
                }
            };
        </script>
    </body>
    </html>
    """


# Backend prediction
@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):  # Accept multiple files
    # Validate that the number of files is between 5 and 10
    if not (5 <= len(files) <= 10):
        return {"error": "Please upload between 5 and 10 images."}

    predictions = []
    

    for file in files:
        # Read the bytes and convert to a NumPy array
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)

        # Decode the image from the NumPy array
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            print("Error loading image.")
        else:
            # Process the image as needed
            # For example, show the image or perform operations
            pass

        image_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Get model predictions
        pred = MODEL.predict(image_batch)

        # Get the predicted class for this image
        predicted_class = CLASS[np.argmax(pred[0])]
        predictions.append(predicted_class)  # Store the class prediction

    # Perform majority voting
    class_count = Counter(predictions)
    majority_class, majority_count = class_count.most_common(1)[0]

    # Calculate confidence (majority count divided by total images)
    confidence = (majority_count / len(predictions)) * 100

    return {"Class": majority_class, "Confidence": confidence}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8018)
