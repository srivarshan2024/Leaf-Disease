from flask import Flask, request, jsonify, render_template
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import and test the prediction module
try:
    from predict import predict_and_recommend, model
    print(f"✓ Prediction module imported successfully")
    print(f"✓ Model loaded: {model is not None}")
except Exception as e:
    print(f"✗ Error importing prediction module: {e}")
    import traceback
    traceback.print_exc()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    # Get class labels from the model
    try:
        from predict import class_labels
        # Ensure class_labels is a list and not None
        if class_labels is None:
            class_labels = []
        return render_template('index.html', class_labels=class_labels)
    except Exception as e:
        print(f"Error getting class labels: {e}")
        return render_template('index.html', class_labels=[])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if not file.filename.lower().split('.')[-1] in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    filepath = None
    try:
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Get prediction and recommendation
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            predict_and_recommend(filepath)
        
        output = f.getvalue().splitlines()
        
        # Clean up the uploaded file
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as cleanup_error:
            print(f"Warning: Could not clean up file {filepath}: {cleanup_error}")
        
        if len(output) >= 2:
            prediction = output[0].replace('Prediction: ', '').strip()
            recommendation = output[1].replace('Recommendation: ', '').strip()
            
            # Validate prediction output
            if not prediction or prediction.startswith('Error:'):
                return jsonify({'error': f'Prediction failed: {prediction}'}), 500
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'recommendation': recommendation
            })
        else:
            error_msg = 'Prediction failed - insufficient output'
            if output:
                error_msg = f'Prediction failed: {output[0]}'
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        # Clean up the uploaded file in case of error
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        print(f"API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)