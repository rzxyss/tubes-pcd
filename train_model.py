import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class SkinTypeClassifier:
    def __init__(self, dataset_path='dataset', img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.classes = ['berjerawat', 'berminyak', 'kering', 'normal']
        self.model = None
        
    def load_and_preprocess_data(self):
        """Load dataset and preprocess images"""
        print("Loading and preprocessing dataset...")
        X = []
        y = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Path {class_path} tidak ditemukan!")
                continue
                
            print(f"Processing class: {class_name}")
            image_count = 0
            
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, filename)
                    
                    # Load image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    # Resize image
                    image = cv2.resize(image, self.img_size)
                    
                    # Normalisasi pixel values
                    image = image.astype('float32') / 255.0
                    
                    X.append(image)
                    y.append(class_idx)
                    image_count += 1
            
            print(f"Loaded {image_count} images from {class_name}")
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Total dataset: {len(X)} images")
        print(f"Image shape: {X[0].shape}")
        
        return X, y
    
    def build_model(self):
        """Build CNN model"""
        model = Sequential([
            # Layer konvolusi, mendeteksi pola dalam gambar (tepi, bentuk, dsb).
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            #  Mengurangi dimensi fitur (downsampling) dengan memilih nilai maksimum di setiap jendela 2x2.
            MaxPooling2D(2, 2),
            
            # Lebih banyak filter untuk mendeteksi fitur yang lebih kompleks
            Conv2D(64, (3, 3), activation='relu'),
            # Mengurangi dimensi fitur (downsampling) dengan memilih nilai maksimum di setiap jendela 2x2.
            MaxPooling2D(2, 2),
            
            # Mendeteksi fitur tingkat tinggi (tekstur kulit, pola jerawat)
            Conv2D(128, (3, 3), activation='relu'),
            # Mengurangi dimensi fitur (downsampling) dengan memilih nilai maksimum di setiap jendela 2x2.
            MaxPooling2D(2, 2),
            
            # Mendeteksi fitur tingkat tinggi (tekstur kulit, pola jerawat)
            Conv2D(128, (3, 3), activation='relu'),
            # Mengurangi dimensi fitur (downsampling) dengan memilih nilai maksimum di setiap jendela 2x2.
            MaxPooling2D(2, 2),
            
            # Flatten and Dense layers
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self):
        """Training model"""
        # Load dan preprocess data
        X, y = self.load_and_preprocess_data()
        
        if len(X) == 0:
            print("Error: Tidak ada data yang berhasil dimuat!")
            return
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, len(self.classes))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Build model
        self.model = self.build_model()
        print(self.model.summary())
        
        # Training
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluasi
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        
        # Prediksi untuk classification report
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=self.classes))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
        # Save model
        self.model.save('skin_type_model.h5')
        print("Model saved as 'skin_type_model.h5'")
        
        # Save class names
        with open('classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)
        print("Class names saved as 'classes.pkl'")

if __name__ == "__main__":
    # Initialize classifier
    classifier = SkinTypeClassifier()
    
    # Train model
    classifier.train()