import argparse
import sys
import os
from models.predict import BreastCancerPredictor

def list_available_models(models_dir):
    """List all available .h5 model files in the models directory"""
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    return available_models

def list_test_images(data_dir):
    """List example test images from different datasets"""
    datasets = ['MIAS', 'CBIS-DDSM', 'INbreast', 'mini-DDSM']
    test_images = {}
    
    for dataset in datasets:
        dataset_path = os.path.join(data_dir, 'processed', dataset, 'test')
        if os.path.exists(dataset_path):
            for category in ['normal', 'benign', 'malignant']:
                category_path = os.path.join(dataset_path, category)
                if os.path.exists(category_path):
                    images = [f for f in os.listdir(category_path) if f.endswith(('.png', '.jpg'))]
                    if images:
                        test_images[f"{dataset}_{category}"] = os.path.join(category_path, images[0])
    
    return test_images

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Breast Cancer Detection CLI')
    parser.add_argument('--image', help='Path to the mammogram image')
    parser.add_argument('--model', help='Name of the model file')
    parser.add_argument('--list-models', action='store_true', help='List available trained models')
    parser.add_argument('--list-images', action='store_true', help='List example test images')
    
    args = parser.parse_args()
    
    # Get base directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    data_dir = os.path.join(base_dir, 'data')
    
    # List available models if requested
    if args.list_models:
        print("\nAvailable trained models:")
        print("-" * 30)
        for model in list_available_models(models_dir):
            print(f"- {model}")
        return
    
    # List example test images if requested
    if args.list_images:
        print("\nExample test images:")
        print("-" * 30)
        for dataset_category, image_path in list_test_images(data_dir).items():
            print(f"\n{dataset_category}:")
            print(f"  {image_path}")
        return
    
    # Check if required arguments are provided
    if not args.image or not args.model:
        print("\nError: Both --image and --model arguments are required.")
        print("\nUsage examples:")
        print("1. List available models:")
        print("   python src/cli.py --list-models")
        print("\n2. List example test images:")
        print("   python src/cli.py --list-images")
        print("\n3. Make prediction:")
        print("   python src/cli.py --image path/to/image.png --model model_name.h5")
        return
    
    # Construct model path
    model_path = os.path.join(models_dir, args.model)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("\nAvailable models:")
        for model in list_available_models(models_dir):
            print(f"- {model}")
        return
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    try:
        # Initialize predictor
        predictor = BreastCancerPredictor(model_path)
        
        # Make prediction
        result = predictor.predict(args.image)
        
        # Print results
        print("\n=== Breast Cancer Detection Results ===")
        print(f"\nImage: {args.image}")
        print(f"Model: {args.model}")
        print(f"\nPredicted Class: {result['class'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        print("\nDetailed Probabilities:")
        print("-" * 30)
        for class_name, prob in result['probabilities'].items():
            print(f"{class_name.title():>10}: {prob:.2%}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()