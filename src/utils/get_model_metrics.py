import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def calculate_all_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics."""
    metrics = {}
    
    # Convert predictions to class labels if needed
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    
    # Calculate metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    try:
        # Calculate precision, recall, and f1 with weighted average
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # For ROC AUC, we need to binarize the labels
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # Use one-vs-rest ROC AUC for multiclass
            metrics['roc_auc'] = roc_auc_score(
                tf.keras.utils.to_categorical(y_true, n_classes),
                y_prob,
                multi_class='ovr',
                average='weighted'
            )
    except Exception as e:
        print(f"Warning: Error calculating some metrics: {str(e)}")
        metrics.setdefault('precision', 0.0)
        metrics.setdefault('recall', 0.0)
        metrics.setdefault('f1_score', 0.0)
        metrics.setdefault('roc_auc', 0.0)
    
    return metrics

def load_test_data(dataset_name):
    # Base path for datasets
    base_path = os.path.join('/Users/saswatkumarsahoo/Desktop/breast-cancer-detection/data/processed', dataset_name)
    test_csv = os.path.join(base_path, 'splits', 'test.csv')
    
    if not os.path.exists(test_csv):
        # Try without splits directory
        test_csv = os.path.join(base_path, 'test.csv')
    
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV file not found at {test_csv}")
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    
    # Handle different column naming conventions
    if 'filepath' in test_df.columns:
        test_df = test_df.rename(columns={'filepath': 'image_path'})
    
    if 'class' in test_df.columns:
        test_df = test_df.rename(columns={'class': 'label'})
    
    # Ensure image paths are absolute
    if not test_df['image_path'].iloc[0].startswith('/'):
        test_df['image_path'] = test_df['image_path'].apply(
            lambda x: os.path.join(base_path, x.replace('\\', '/'))
        )
    
    # Create data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create test generator
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=None,  # Since we're using absolute paths
        x_col="image_path",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

def evaluate_model(model_path, dataset_name):
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        model = load_model(model_path)
        
        # Load test data
        test_generator = load_test_data(dataset_name)
        
        # Get predictions
        y_prob = model.predict(test_generator)
        y_test = test_generator.labels
        
        # Convert predictions to class labels
        y_pred = np.argmax(y_prob, axis=1)
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_test, y_pred, y_prob)
        
        return metrics
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    base_dir = '/Users/saswatkumarsahoo/Desktop/breast-cancer-detection'
    datasets = ['CBIS-DDSM', 'INbreast', 'MIAS', 'mini-DDSM']
    
    for dataset in datasets:
        print(f"\nMetrics for {dataset} dataset:")
        model_filename_map = {
            'CBIS-DDSM': 'cbis_ddsm_model_best.h5',
            'INbreast': 'inbreast_model_best.h5',
            'MIAS': 'mias_model.h5',
            'mini-DDSM': 'mini_ddsm_model_best.h5'
        }
        
        model_path = os.path.join(base_dir, 'models', model_filename_map[dataset])
        
        try:
            metrics = evaluate_model(model_path, dataset)
            if metrics:
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-score: {metrics['f1_score']:.4f}")
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        except Exception as e:
            print(f"Error evaluating {dataset} model: {str(e)}")