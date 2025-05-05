import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from PIL import Image
import numpy as np

# Set plot style
plt.style.use('seaborn-v0_8')  
sns.set_style("whitegrid")  


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


def show_sample_images(source, num_samples=5):
    """
    Display sample images from each emotion category in a directory,
    or from a provided list of image paths.
    
    Args:
        source (str or list): Path to the dataset directory OR a list of image file paths.
        num_samples (int): Number of images to show per category or from the list.
    """
    if isinstance(source, str):
        # Assume it's a directory: show samples from each subfolder
        emotion_dirs = os.listdir(source)
        for emotion in emotion_dirs:
            emotion_path = os.path.join(source, emotion)
            if os.path.isdir(emotion_path):
                print(f"\n{emotion.upper()}:")
                image_files = glob.glob(os.path.join(emotion_path, '*.jpg'))[:num_samples]
                plt.figure(figsize=(3 * num_samples, 5))
                for idx, img_path in enumerate(image_files):
                    try:
                        img = Image.open(img_path)
                        plt.subplot(1, num_samples, idx + 1)
                        plt.imshow(img)
                        plt.axis('off')
                    except Exception as e:
                        print(f"Error displaying {img_path}: {e}")
                plt.show()
    elif isinstance(source, list):
        # Assume it's a list of image paths
        print(f"\nDisplaying {min(num_samples, len(source))} images from provided list:")
        plt.figure(figsize=(3 * num_samples, 5))
        for idx, img_path in enumerate(source[:num_samples]):
            try:
                img = Image.open(img_path)
                plt.subplot(1, num_samples, idx + 1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(os.path.basename(img_path), fontsize=8)
            except Exception as e:
                print(f"Error displaying {img_path}: {e}")
        plt.show()
    else:
        raise ValueError("source must be a directory path (str) or a list of image paths.")


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


def analyze_color_distribution(path: str, num_samples: int = 100) -> None:
    """
    Analyze and plot color distribution (RGB histograms) for images in the dataset.
    Args:
        path (str): Path to the dataset directory
        num_samples (int): Number of images to sample per emotion category
    """
    # Initialize accumulators for RGB histograms
    hist_accum = [np.zeros(256), np.zeros(256), np.zeros(256)]
    total_images = 0
    
    for emotion in os.listdir(path):
        emotion_path = os.path.join(path, emotion)
        if os.path.isdir(emotion_path):
            # Sample images from each emotion category
            image_files = glob.glob(os.path.join(emotion_path, '*.jpg'))[:num_samples]
            
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_np = np.array(img)
                    
                    # Calculate histograms for each channel
                    for i in range(3):  # R, G, B
                        hist, _ = np.histogram(img_np[:,:,i].ravel(), bins=256, range=(0,256))
                        hist_accum[i] += hist
                    
                    total_images += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Plot the accumulated histograms
    plt.figure(figsize=(15, 5))
    colors = ['Red', 'Green', 'Blue']
    
    for i, color in enumerate(colors):
        plt.subplot(1, 3, i + 1)
        plt.bar(np.arange(256), hist_accum[i] / total_images, color=color.lower(), alpha=0.7)
        plt.title(f'{color} Channel Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Normalized Frequency')
    
    plt.tight_layout()
    plt.show()


def analyze_pixel_distribution(path: str, num_samples: int = 50) -> None:
    """
    Analyze and plot pixel value distribution across the dataset.
    Args:
        path (str): Path to the dataset directory
        num_samples (int): Number of images to sample per emotion category
    """
    pixel_values = []
    
    for emotion in os.listdir(path):
        emotion_path = os.path.join(path, emotion)
        if os.path.isdir(emotion_path):
            # Sample images from each emotion category
            image_files = glob.glob(os.path.join(emotion_path, '*.jpg'))[:num_samples]
            
            for img_path in image_files:
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img_np = np.array(img)
                    pixel_values.extend(img_np.ravel())
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Plot pixel value distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(pixel_values, bins=256, kde=True)
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    
    # Print statistics
    print("Pixel Value Statistics:")
    print(f"Mean: {np.mean(pixel_values):.2f}")
    print(f"Standard Deviation: {np.std(pixel_values):.2f}")
    print(f"Min: {np.min(pixel_values)}")
    print(f"Max: {np.max(pixel_values)}")


def combine_emotion_classes(df: pd.DataFrame, source_class: str, target_class: str) -> pd.DataFrame:
    """
    Combine two emotion classes in a DataFrame and return the updated DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing emotion classes and their counts
        source_class (str): The class to be merged (e.g., 'disgust')
        target_class (str): The class to merge into (e.g., 'angry')
    Returns:
        pd.DataFrame: Updated DataFrame with combined classes
    """
    # Create a copy to avoid modifying the original
    df_combined = df.copy()
    
    # Get the count of the source class
    source_count = df_combined.loc[df_combined['Emotion'] == source_class, 'Count'].sum()
    
    # Add the count to the target class
    df_combined.loc[df_combined['Emotion'] == target_class, 'Count'] += source_count
    
    # Remove the source class
    df_combined = df_combined[df_combined['Emotion'] != source_class]
    
    return df_combined


def plot_class_comparison(original_df: pd.DataFrame, combined_df: pd.DataFrame, title: str) -> None:
    """
    Plot a comparison between original and combined class distributions.
    Args:
        original_df (pd.DataFrame): Original DataFrame with emotion classes
        combined_df (pd.DataFrame): DataFrame with combined emotion classes
        title (str): Title for the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original distribution
    sns.barplot(x='Emotion', y='Count', data=original_df, ax=ax1)
    ax1.set_title('Original Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot combined distribution
    sns.barplot(x='Emotion', y='Count', data=combined_df, ax=ax2)
    ax2.set_title('After Combining Classes')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show() 

def find_blank_images(folder_path, threshold=5):
    """
    Returns a list of image file paths that are likely blank or corrupted.
    Args:
        folder_path (str): Path to the folder containing images.
        threshold (int): Minimum standard deviation to consider an image as non-blank.
    Returns:
        List of file paths.
    """
    blank_images = []
    for img_file in glob.glob(os.path.join(folder_path, '*.jpg')):
        try:
            img = Image.open(img_file).convert('L')
            img_np = np.array(img)
            if img_np.std() < threshold:
                blank_images.append(img_file)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    return blank_images

def count_images_in_folder(base_path):
    """
    Count the total number of .jpg images in all subfolders of a base directory.
    Args:
        base_path (str): Path to the base directory (e.g., train or test folder)
    Returns:
        int: Total number of images
    """
    total = 0
    for emotion in os.listdir(base_path):
        emotion_path = os.path.join(base_path, emotion)
        if os.path.isdir(emotion_path):
            total += len(glob.glob(os.path.join(emotion_path, '*.jpg')))
    return total