# Plant Disease Detection System

A web-based application that uses deep learning to detect plant diseases from leaf images. The system can identify 15 different plant diseases across multiple crops including Apple, Cherry, Corn, Grape, and Peach.

## Features

- **Real-time Disease Detection**: Upload leaf images and get instant disease predictions
- **15 Disease Classes**: Supports detection of various diseases across 5 different crops
- **Treatment Recommendations**: Provides specific treatment recommendations for each detected disease
- **Modern Web Interface**: Beautiful, responsive web interface built with HTML, CSS, and JavaScript
- **RESTful API**: Backend API for easy integration

## Supported Crops and Diseases

### Apple
- Black Rot
- Cedar Apple Rust
- Healthy

### Cherry
- Powdery Mildew
- Healthy

### Corn
- Cercospora Leaf Spot
- Common Rust
- Northern Leaf Blight
- Healthy

### Grape
- Black Rot
- Esca (Black Measles)
- Leaf Blight
- Healthy

### Peach
- Bacterial Spot
- Healthy

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Plant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the model file**
   Ensure `plant_leaf_diseases_model.h5` is in the root directory.

## Usage

### Running the Web Application

1. **Start the Flask server**
   ```bash
   python app/app.py
   ```

2. **Open your browser**
   Navigate to `http://localhost:5000`

3. **Upload an image**
   - Click "Choose Leaf Image" to select a plant leaf image
   - Click "Analyze" to get the disease prediction and treatment recommendation

### API Usage

The application provides a REST API for programmatic access:

**Endpoint**: `POST /api/predict`

**Request**: Multipart form data with image file

**Response**:
```json
{
  "success": true,
  "prediction": "Apple__Black_rot",
  "recommendation": "Apply Copper-based fungicide. Dosage: 40g/tree."
}
```

**Example using curl**:
```bash
curl -X POST -F "file=@path/to/leaf_image.jpg" http://localhost:5000/api/predict
```

## Testing

### Test the Model
```bash
python test_model.py
```

### Test the API
```bash
python test_flask_api.py
```

## Technical Details

- **Model**: Convolutional Neural Network (CNN) trained on plant disease dataset
- **Input Size**: 256x256 pixels
- **Framework**: TensorFlow/Keras
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: TensorFlow Image preprocessing

## File Structure

```
Plant/
├── app/
│   ├── app.py              # Flask application
│   └── templates/
│       └── index.html      # Web interface
├── src/
│   ├── predict.py          # Prediction logic
│   └── train.py            # Training script
├── data/                   # Training dataset
├── uploads/                # Temporary upload directory
├── plant_leaf_diseases_model.h5  # Trained model
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Performance

The model has been trained on a comprehensive dataset of plant disease images and can accurately classify:
- 15 different disease classes
- 5 different crop types
- Healthy vs diseased plants

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on the repository.