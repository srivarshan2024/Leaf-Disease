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

# Mandatory class labels based on your dataset
class_labels = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

fertilizer_map = {}

def generate_fertilizer_recommendations(class_labels):
    """Generate fertilizer recommendations based on class labels"""
    recommendations = {}
    
    for label in class_labels:
        label_lower = label.lower()
        
        if 'healthy' in label_lower:
            recommendations[label] = f'No treatment needed. Plant is healthy. Continue regular maintenance.'
        elif 'bacterial' in label_lower:
            recommendations[label] = 'Apply Copper-based bactericide. Dosage: 30g per plant.'
        elif 'blight' in label_lower:
            recommendations[label] = 'Use Mancozeb or Chlorothalonil fungicide. Dosage: 40g per plant.'
        elif 'leaf_mold' in label_lower or 'leaf' in label_lower:
            recommendations[label] = 'Apply Sulfur-based fungicide. Dosage: 25g per plant.'
        elif 'spider_mites' in label_lower:
            recommendations[label] = 'Use insecticidal soap or neem oil. Dosage: as per instructions.'
        elif 'target_spot' in label_lower:
            recommendations[label] = 'Apply Azoxystrobin fungicide. Dosage: 50g per acre.'
        elif 'yellowleaf' in label_lower or 'curl_virus' in label_lower or 'mosaic_virus' in label_lower:
            recommendations[label] = 'No chemical treatment. Remove infected plants. Use resistant varieties.'
        else:
            recommendations[label] = f'Apply general treatment for {label}. Consult with agricultural expert for specific dosage.'
    
    return recommendations

# Load model if it exists
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded successfully with {len(class_labels)} classes.")
        print(f"Class labels: {class_labels}")
        fertilizer_map = generate_fertilizer_recommendations(class_labels)
        print(f"Generated {len(fertilizer_map)} fertilizer recommendations.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print("No trained model found. Running in demo mode.")

def predict_and_recommend(img_path):
    try:
        if model is None:
            # Demo mode
            import random
            prediction = random.choice(class_labels)
            recommendation = fertilizer_map.get(prediction, 'No recommendation available.')
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
        preds = model.predict(x, verbose=0)
        class_idx = np.argmax(preds, axis=1)[0]

        if class_idx >= len(class_labels):
            print(f"Error: Predicted class index {class_idx} is out of bounds.")
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
