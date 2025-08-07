import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(current_dir), 'plant_leaf_diseases_model.h5')
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Load model if it exists, otherwise use demo mode
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("No trained model found. Running in demo mode.")

# Class labels - these should match the model's output classes (15 classes)
class_labels = [
    'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
    'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn__Cercospora_leaf_spot Gray_leaf_spot', 'Corn__Common_rust', 'Corn__healthy', 'Corn__Northern_Leaf_Blight',
    'Grape__Black_rot', 'Grape__Esca_(Black_Measles)', 'Grape__healthy', 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Peach___Bacterial_spot', 'Peach___healthy'
]

# Fertilizer recommendation mapping (updated for all 15 classes)
fertilizer_map = {
    'Apple__Black_rot': 'Apply Copper-based fungicide. Dosage: 40g/tree.',
    'Apple__Cedar_apple_rust': 'Apply Myclobutanil fungicide. Dosage: 35g/tree.',
    'Apple__healthy': 'No treatment needed. Apple tree is healthy. Continue regular maintenance.',
    'Cherry___Powdery_mildew': 'Apply Sulfur-based fungicide. Dosage: 25g/tree.',
    'Cherry___healthy': 'No treatment needed. Cherry tree is healthy. Continue regular maintenance.',
    'Corn__Cercospora_leaf_spot Gray_leaf_spot': 'Apply Azoxystrobin fungicide. Dosage: 50g/acre.',
    'Corn__Common_rust': 'Apply Pyraclostrobin fungicide. Dosage: 45g/acre.',
    'Corn__healthy': 'No treatment needed. Corn is healthy. Continue regular maintenance.',
    'Corn__Northern_Leaf_Blight': 'Apply Mancozeb fungicide. Dosage: 55g/acre.',
    'Grape__Black_rot': 'Apply Mancozeb fungicide. Dosage: 30g/vine.',
    'Grape__Esca_(Black_Measles)': 'Apply Tebuconazole fungicide. Dosage: 35g/vine.',
    'Grape__healthy': 'No treatment needed. Grape vine is healthy. Continue regular maintenance.',
    'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply Copper-based fungicide. Dosage: 40g/vine.',
    'Peach___Bacterial_spot': 'Apply Copper-based bactericide. Dosage: 30g/tree.',
    'Peach___healthy': 'No treatment needed. Peach tree is healthy. Continue regular maintenance.'
}

def predict_and_recommend(img_path):
    try:
        if model is None:
            # Demo mode - return sample predictions
            import random
            demo_predictions = ['Apple__healthy', 'Cherry___healthy', 'Grape__healthy', 'Corn__healthy']
            demo_recommendations = [
                'No treatment needed. Apple tree is healthy. Continue regular maintenance.',
                'No treatment needed. Cherry tree is healthy. Continue regular maintenance.',
                'No treatment needed. Grape vine is healthy. Continue regular maintenance.',
                'No treatment needed. Corn is healthy. Continue regular maintenance.'
            ]
            prediction = random.choice(demo_predictions)
            recommendation = random.choice(demo_recommendations)
            print(f"Prediction: {prediction}")
            print(f"Recommendation: {recommendation}")
            return
        
        # Check if image file exists
        if not os.path.exists(img_path):
            print(f"Error: Image file not found at {img_path}")
            return
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        
        # Make prediction
        preds = model.predict(x, verbose=0)  # Suppress verbose output
        class_idx = np.argmax(preds, axis=1)[0]
        
        # Ensure class_idx is within bounds
        if class_idx >= len(class_labels):
            print(f"Error: Predicted class index {class_idx} is out of bounds for {len(class_labels)} classes")
            return
        
        class_name = class_labels[class_idx]
        recommendation = fertilizer_map.get(class_name, 'No recommendation available.')
        
        print(f"Prediction: {class_name}")
        print(f"Recommendation: {recommendation}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_and_recommend(sys.argv[1])