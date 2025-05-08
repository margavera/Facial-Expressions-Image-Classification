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

def build_image_emotion_df(base_path):
    """
    Build a DataFrame with columns: 'filename' and 'Emotion' for all images in base_path.
    Args:
        base_path (str): Path to the dataset directory (train or test)
    Returns:
        pd.DataFrame: DataFrame with columns 'filename' and 'Emotion'
    """
    data = []
    for emotion in os.listdir(base_path):
        emotion_path = os.path.join(base_path, emotion)
        if os.path.isdir(emotion_path):
            for img_path in glob.glob(os.path.join(emotion_path, '*.jpg')):
                filename = os.path.basename(img_path)
                data.append({'filename': filename, 'Original Emotion': emotion})
    return pd.DataFrame(data)

def plot_class_distribution(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """
    Plots the distribution of images across different categories.
    Args:
        df (pd.DataFrame): DataFrame containing the data to plot
        x_col (str): Column name for the x-axis
        y_col (str): Column name for the y-axis
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x=x_col, y=y_col, data=df)
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

def analyze_image_dimensions(image_paths):
    """
    Analyze image dimensions for a given list of image file paths.
    Args:
        image_paths (list): List of image file paths.
    """
    dimensions = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                dimensions.append(img.size)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not dimensions:
        print("No valid images found.")
        return

    dim_df = pd.DataFrame(dimensions, columns=['Width', 'Height'])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(dim_df['Width'], kde=True)
    plt.title('Width Distribution')

    plt.subplot(1, 2, 2)
    sns.histplot(dim_df['Height'], kde=True)
    plt.title('Height Distribution')

    plt.tight_layout()
    plt.show()

    print("Image Dimension Statistics:")
    print(dim_df.describe())

def analyze_color_distribution(image_paths):
    """
    Analyze and plot color distribution (RGB histograms) for a list of image file paths.
    Args:
        image_paths (list): List of image file paths.
    """
    hist_accum = [np.zeros(256), np.zeros(256), np.zeros(256)]
    total_images = 0

    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            for i in range(3):  # R, G, B
                hist, _ = np.histogram(img_np[:,:,i].ravel(), bins=256, range=(0,256))
                hist_accum[i] += hist
            total_images += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if total_images == 0:
        print("No valid images found.")
        return

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

def analyze_pixel_distribution(image_paths):
    """
    Analyze and plot pixel value distribution for a list of image file paths.
    Args:
        image_paths (list): List of image file paths.
    """
    pixel_values = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_np = np.array(img)
            pixel_values.extend(img_np.ravel())
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not pixel_values:
        print("No valid images found.")
        return

    plt.figure(figsize=(12, 6))
    sns.histplot(pixel_values, bins=256, kde=True)
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

    print("Pixel Value Statistics:")
    print(f"Mean: {np.mean(pixel_values):.2f}")
    print(f"Standard Deviation: {np.std(pixel_values):.2f}")
    print(f"Min: {np.min(pixel_values)}")
    print(f"Max: {np.max(pixel_values)}")

def plot_class_comparison(original_df: pd.DataFrame, combined_df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    """
    Plot a comparison between original and combined class distributions.
    Args:
        original_df (pd.DataFrame): Original DataFrame with emotion classes
        combined_df (pd.DataFrame): DataFrame with combined emotion classes
        x_col (str): Column name for the x-axis
        y_col (str): Column name for the y-axis
        title (str): Title for the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original distribution
    sns.barplot(x=x_col, y=y_col, data=original_df, ax=ax1)
    ax1.set_title('Original Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot combined distribution
    sns.barplot(x=x_col, y=y_col, data=combined_df, ax=ax2)
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

def get_image_paths(df, base_path, filename_col='filename', emotion_col='Emotion'):
    """
    Given a DataFrame with columns for filename and emotion, return a list of full, normalized image paths.
    
    Args:
        df (pd.DataFrame): DataFrame containing the image data (e.g., 'filename' and 'Emotion' columns).
        base_path (str): The base directory where images are stored.
        filename_col (str): The name of the column containing image filenames.
        emotion_col (str): The name of the column containing emotion labels (subdirectories).
        
    Returns:
        list: A list of full, normalized image paths.
    """
    import os
    return [
        os.path.normpath(os.path.join(base_path, row[emotion_col], row[filename_col]))
        for _, row in df.iterrows()
    ]

def balance_classes(df, target_size, class_col='Emotion', filename_col='filename', random_state=42):
    """
    Downsample each class in the DataFrame to achieve a target size while keeping the classes balanced.
    
    Args:
        df (pd.DataFrame): DataFrame with at least the column for class labels (e.g., 'Emotion') and filenames (e.g., 'filename').
        target_size (int): The target number of images in the resulting dataset.
        class_col (str): The name of the column containing the class labels (default is 'Emotion').
        filename_col (str): The name of the column containing the filenames (default is 'filename').
        random_state (int): For reproducibility.
        
    Returns:
        pd.DataFrame: Balanced DataFrame with approximately the target size.
    """
    # Number of unique classes
    n_classes = df[class_col].nunique()

    # Proportional reduction in each class
    class_counts = df[class_col].value_counts()

    # Calculate proportional number of samples per class
    total_samples_per_class = int(target_size / n_classes)
    
    # Ensure we don't sample more than available images in each class
    desired_samples_per_class = {
        class_name: min(total_samples_per_class, count) 
        for class_name, count in class_counts.items()
    }

    # Downsample each class according to the desired samples per class
    balanced_df = df.groupby(class_col, group_keys=False).apply(
        lambda group: group.sample(desired_samples_per_class[group[class_col].iloc[0]], random_state=random_state)
    ).reset_index(drop=True)

    return balanced_df


