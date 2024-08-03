import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st

# Load the trained model
model = load_model('leaf_disease_model.h5')

# Get class labels from the training directory

labels = {'Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'}

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
