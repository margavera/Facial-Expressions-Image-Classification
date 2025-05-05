import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import glob
from PIL import Image
import numpy as np


def get_class_distribution(path: str) -> dict:
    """
    Get the distribution of images across emotion classes.
    Args:
        path (str): Path to the dataset directory (train or test)
    Returns:
        dict: Dictionary with emotion categories as keys and number of images as values
    """
    emotion_dirs = os.listdir(path)
    distribution = {}
    
    for emotion in emotion_dirs:
        emotion_path = os.path.join(path, emotion)
        if os.path.isdir(emotion_path):
            num_images = len(os.listdir(emotion_path))
            distribution[emotion] = num_images
    
    return distribution


def plot_class_distribution(df: pd.DataFrame, title: str) -> None:
    """
    Plots the distribution of images across emotion categories.
    Args:
        df (pd.DataFrame): DataFrame containing emotion categories and their counts
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Emotion', y='Count', data=df)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_sample_images(path: str, num_samples: int = 5) -> None:
    """
    Display sample images from each emotion category.
    Args:
        path (str): Path to the dataset directory
        num_samples (int): Number of sample images to show per category
    """
    emotion_dirs = os.listdir(path)
    
    for emotion in emotion_dirs:
        emotion_path = os.path.join(path, emotion)
        if os.path.isdir(emotion_path):
            print(f"\n{emotion.upper()}:")
            
            # Get sample images
            image_files = glob.glob(os.path.join(emotion_path, '*.jpg'))[:num_samples]
            
            plt.figure(figsize=(15, 3))
            for idx, img_path in enumerate(image_files):
                img = Image.open(img_path)
                plt.subplot(1, num_samples, idx + 1)
                plt.imshow(img)
                plt.axis('off')
            plt.show()


def analyze_image_dimensions(path: str) -> None:
    """
    Analyze image dimensions across the dataset.
    Args:
        path (str): Path to the dataset directory
    """
    dimensions = []
    
    for emotion in os.listdir(path):
        emotion_path = os.path.join(path, emotion)
        if os.path.isdir(emotion_path):
            for img_file in os.listdir(emotion_path)[:100]:  # Sample 100 images per category
                img_path = os.path.join(emotion_path, img_file)
                with Image.open(img_path) as img:
                    dimensions.append(img.size)
    
    # Convert to DataFrame
    dim_df = pd.DataFrame(dimensions, columns=['Width', 'Height'])
    
    # Plot distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(dim_df['Width'], kde=True)
    plt.title('Width Distribution')
    
    plt.subplot(1, 2, 2)
    sns.histplot(dim_df['Height'], kde=True)
    plt.title('Height Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("Image Dimension Statistics:")
    print(dim_df.describe()) 