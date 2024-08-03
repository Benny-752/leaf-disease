import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st

# Load the trained model
model = load_model('leaf_disease_model.h5')

# Get class labels from the training directory
train_dir = r'C:\Users\Benny Solomon\Downloads\dataset\train'
labels = {i: label for i, label in enumerate(os.listdir(train_dir))}

# Function to get prediction results
def getResult(image_path):
    img = image.load_img(image_path, target_size=(128, 128))  # Adjust to your model's input shape
    x = image.img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

# Streamlit app
st.title("Leaf Disease Detection")

st.write("""
### Upload a leaf image to get disease prediction
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to disk
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    st.write("")
    st.write("Classifying...")
    
    # Get prediction results
    predictions = getResult(file_path)
    predicted_label = labels[np.argmax(predictions)]
    predicted_probability = np.max(predictions)
    
    st.write(f"**Prediction:** {predicted_label}")
    st.write(f"**Probability:** {predicted_probability:.2f}")

if not os.path.exists('uploads'):
    os.makedirs('uploads')
