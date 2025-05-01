import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def quote_product_display_name(input_path: str, output_path: str) -> None:
    """
    Processes a CSV file so that after the first 9 commas in each line, the rest of the line is enclosed in quotes.
    This is useful for cleaning productDisplayName fields that may contain commas, preventing CSV parsing errors.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            comma_indices = [i for i, c in enumerate(line) if c == ',']
            if len(comma_indices) > 9:
                ninth_comma = comma_indices[8]
                before = line[:ninth_comma+1]
                after = line[ninth_comma+1:].strip()
                if after.startswith('"') and after.endswith('"'):
                    outfile.write(line)
                else:
                    outfile.write(before + '"' + after + '"\n')
            else:
                outfile.write(line)


def merge_data(styles_df: pd.DataFrame, images_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the styles and images DataFrames on the 'id' column and returns the merged DataFrame.
    """
    # Ensure 'id' column is present and of the correct type in both DataFrames
    if 'id' not in images_df.columns:
        images_df['id'] = images_df['filename'].str.replace('.jpg', '', regex=False).astype(int)
    if 'id' in styles_df.columns:
        styles_df['id'] = styles_df['id'].astype(int)
    df_merge = pd.merge(styles_df, images_df, on='id', how='inner')
    return df_merge


def plot_class_distributions(df: pd.DataFrame, variables: list, top_n: int = 10) -> None:
    """
    Plots count distributions for the specified categorical variables.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        variables (list): List of column names (categorical variables) to plot.
        top_n (int): For variables with many categories, plot only the top_n most frequent.
    """
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(variables)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        # Si hay muchas categorías, solo mostramos las top_n
        order = df[var].value_counts().index[:top_n] if df[var].nunique() > top_n else df[var].value_counts().index
        sns.countplot(data=df, x=var, ax=axes[i], order=order)
        axes[i].set_title(f'Distribution of {var}')
        axes[i].tick_params(axis='x', rotation=45)
    # Elimina ejes vacíos si hay
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def plot_year_season_distribution(df: pd.DataFrame) -> None:
    """
    Plots the distribution of products over years and seasons.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x='year', ax=axes[0], order=sorted(df['year'].dropna().unique()))
    axes[0].set_title('Year Distribution')
    sns.countplot(data=df, x='season', ax=axes[1], order=df['season'].value_counts().index)
    axes[1].set_title('Season Distribution')
    plt.tight_layout()
    plt.show()


def missing_values_summary(df: pd.DataFrame) -> pd.Series:
    """
    Returns a summary of missing values per column in the DataFrame.
    """
    return df.isnull().sum()


def plot_dataset_histogram(image_paths, sample_size=100):
    """
    Plots the accumulated log-scale color histograms (R, G, B) for a sample of images.
    Args:
        image_paths (list): List of image file paths.
        sample_size (int): Number of images to sample and analyze.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import random

    # Sample images
    sample_files = random.sample(image_paths, min(sample_size, len(image_paths)))
    hist_accum = [np.zeros(256), np.zeros(256), np.zeros(256)]

    for img_path in sample_files:
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            for i in range(3):
                hist, _ = np.histogram(img_np[:,:,i].ravel(), bins=256, range=(0,256))
                hist_accum[i] += hist
        except Exception as e:
            continue

    plt.figure(figsize=(12,4))
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        plt.subplot(1,3,i+1)
        plt.bar(np.arange(256), hist_accum[i], color=color.lower(), alpha=0.7)
        plt.yscale('log')
        plt.title(f'Accumulated Histogram {color}')
    plt.tight_layout()
    plt.show()


def show_sample_images(image_paths, sample_size=5):
    """
    Displays a row of sample images from the provided list of image file paths.
    Args:
        image_paths (list): List of image file paths.
        sample_size (int): Number of images to display (default 5).
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    import random

    sample_paths = random.sample(image_paths, min(sample_size, len(image_paths)))

    plt.figure(figsize=(3 * sample_size, 5))
    for i, img_path in enumerate(sample_paths):
        try:
            img = Image.open(img_path)
            plt.subplot(1, sample_size, i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(img_path))
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    plt.tight_layout()
    plt.show() 