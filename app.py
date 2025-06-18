from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import base64
import io
from PIL import Image
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

class SkinTypePredictor:
    def __init__(self):
        self.model = None
        self.classes = None
        self.img_size = (224, 224)
        self.load_model_and_classes()
    
    def load_model_and_classes(self):
        """Load trained model dan class names"""
        try:
            self.model = load_model('skin_type_model.h5')
            with open('classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)
            print("Model dan classes berhasil dimuat!")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def histogram_equalization(self, image):
        """Melakukan histogram equalization pada gambar"""
        if len(image.shape) == 3:
            # Konversi BGR ke YUV
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            # Equalization pada channel Y (luminance)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            # Konversi kembali ke BGR
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            equalized = cv2.equalizeHist(image)
        return equalized
    
    def apply_dft(self, image):
        """Menerapkan DFT pada gambar dan mengembalikan magnitude spectrum"""
        if len(image.shape) == 3:
            # Convert to grayscale first
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply DFT
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
        
        # Normalize to 0-255
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_spectrum = np.uint8(magnitude_spectrum)
        
        # Apply color map for better visualization
        magnitude_spectrum_color = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_JET)
        
        return magnitude_spectrum_color
    
    def extract_dft_features(self, image):
        """Mengekstrak fitur DFT dari gambar"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size for consistent feature extraction
        gray = cv2.resize(gray, (224, 224))
        
        # Apply DFT
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        
        # Get magnitude spectrum
        magnitude_spectrum = np.abs(dft_shift)
        
        # Create a circular mask for frequency bands
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Extract features from different frequency bands
        features = {}
        
        # Low frequencies (center)
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), 30, 1, -1)
        low_freq = magnitude_spectrum * mask
        features['low_freq_mean'] = np.mean(low_freq)
        features['low_freq_std'] = np.std(low_freq)
        
        # Medium frequencies
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), 60, 1, -1)
        cv2.circle(mask, (ccol, crow), 30, 0, -1)
        mid_freq = magnitude_spectrum * mask
        features['mid_freq_mean'] = np.mean(mid_freq)
        features['mid_freq_std'] = np.std(mid_freq)
        
        # High frequencies (outer)
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), 60, 0, -1)
        high_freq = magnitude_spectrum * (1 - mask)
        features['high_freq_mean'] = np.mean(high_freq)
        features['high_freq_std'] = np.std(high_freq)
        
        return features
    
    def segment_skin(self, image):
        """Segmentasi kulit menggunakan color thresholding"""
        # Convert to YCrCb color space (better for skin detection)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create skin mask
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Apply mask to original image
        segmented = cv2.bitwise_and(image, image, mask=mask)
        
        return segmented, mask
    
    def preprocess_image(self, image):
        """Preprocessing gambar untuk prediksi"""
        # Resize image
        image_resized = cv2.resize(image, self.img_size)
        
        # Apply histogram equalization
        image_eq = self.histogram_equalization(image_resized)
        
        # Normalisasi
        image_normalized = image_eq.astype('float32') / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, image_eq
    
    def predict(self, image):
        """Melakukan prediksi pada gambar"""
        if self.model is None:
            return None, None, None
        
        # Preprocess
        processed_image, equalized_image = self.preprocess_image(image)
        
        # Extract DFT features
        dft_features = self.extract_dft_features(image)
        
        # Segment skin
        segmented_image, skin_mask = self.segment_skin(image)
        
        # Prediksi
        predictions = self.model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.classes[predicted_class_idx]
        
        # Get all probabilities
        all_predictions = {}
        for i, class_name in enumerate(self.classes):
            all_predictions[class_name] = float(predictions[0][i])
        
        # Apply DFT for visualization
        dft_image = self.apply_dft(image)
        
        return predicted_class, confidence, all_predictions, equalized_image, dft_image, dft_features, segmented_image, skin_mask

# Initialize predictor
predictor = SkinTypePredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Predict
        predicted_class, confidence, all_predictions, equalized_image, dft_image, dft_features, segmented_image, skin_mask = predictor.predict(image)
        
        if predicted_class is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Convert images to base64 for display
        def image_to_base64(img):
            _, buffer = cv2.imencode('.png', img)
            return base64.b64encode(buffer).decode('utf-8')
        
        equalized_b64 = image_to_base64(equalized_image)
        original_b64 = image_to_base64(cv2.resize(image, (224, 224)))
        dft_b64 = image_to_base64(dft_image)
        segmented_b64 = image_to_base64(segmented_image)
        mask_b64 = image_to_base64(skin_mask)
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'equalized_image': equalized_b64,
            'original_image': original_b64,
            'dft_image': dft_b64,
            'segmented_image': segmented_b64,
            'skin_mask': mask_b64,
            'dft_features': dft_features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    model_loaded = predictor.model is not None
    classes_loaded = predictor.classes is not None
    return jsonify({
        'status': 'healthy' if model_loaded and classes_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'classes_loaded': classes_loaded,
        'available_classes': predictor.classes if classes_loaded else None
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=5000)