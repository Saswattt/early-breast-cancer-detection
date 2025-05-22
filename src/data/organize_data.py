import os
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import shutil

def organize_data():
    # Base paths
    base_dir = '/Users/saswatkumarsahoo/Desktop/breast-cancer-detection/data/processed/mias'
    
    # Create category directories
    categories = ['normal', 'benign', 'cancer']
    for category in categories:
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)
    
    # Look for images with various extensions
    image_extensions = ['*.pgm', '*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(base_dir, ext)))
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {base_dir}")
    
    print(f"Found {len(image_files)} images")
    
    # Create dataframe from image files
    data = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # Determine category and move file
        if 'normal' in filename.lower():
            category = 'normal'
            label = 0
        elif 'benign' in filename.lower():
            category = 'benign'
            label = 1
        else:
            category = 'cancer'
            label = 2
        
        # Move file to category directory
        new_path = os.path.join(base_dir, category, filename)
        if img_path != new_path:
            shutil.move(img_path, new_path)
        
        # Store relative path
        rel_path = os.path.join(category, filename)
        data.append([rel_path, label])
    
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    
    # Print label distribution
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Save processed data
    processed_data_path = os.path.join(base_dir, 'processed_data.csv')
    df.to_csv(processed_data_path, index=False)
    print(f"\nCreated processed data file with {len(df)} images")
    
    # Split the data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # Save the splits
    train_df.to_csv(os.path.join(base_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(base_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(base_dir, 'test.csv'), index=False)
    
    print(f"\nData split complete:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    organize_data()