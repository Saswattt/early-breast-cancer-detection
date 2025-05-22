import tensorflow as tf
import numpy as np
import os

class BreastCancerPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = 224
        self.class_names = ['normal', 'benign', 'malignant']
    
    def preprocess_image(self, image_path):
        # Read image
        img = tf.io.read_file(image_path)
        # Decode PNG format
        img = tf.image.decode_png(img, channels=3)
        # Resize
        img = tf.image.resize(img, [self.img_size, self.img_size])
        # Normalize
        img = img / 255.0
        # Add batch dimension
        img = tf.expand_dims(img, 0)
        return img
    
    def predict(self, image_path):
        # Preprocess image
        processed_img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, predictions[0])
            }
        }

def main():
    # Example usage
    model_path = '/Users/saswatkumarsahoo/Desktop/breast-cancer-detection/models/inbreast_model_best.h5'
    predictor = BreastCancerPredictor(model_path)
    
    # Example prediction
    image_path = 'path/to/your/test/image.png'
    if os.path.exists(image_path):
        result = predictor.predict(image_path)
        print("\nPrediction Results:")
        print(f"Predicted Class: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"{class_name}: {prob:.2%}")
    else:
        print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main()