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


def plot_class_distributions(df: pd.DataFrame) -> None:
    """
    Plots distributions for gender, masterCategory, top 10 subCategories, and top 10 baseColours.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sns.countplot(data=df, x='gender', ax=axes[0,0])
    axes[0,0].set_title('Gender Distribution')
    sns.countplot(data=df, x='masterCategory', ax=axes[0,1])
    axes[0,1].set_title('Master Category Distribution')
    sns.countplot(data=df, x='subCategory', ax=axes[1,0], order=df['subCategory'].value_counts().index[:10])
    axes[1,0].set_title('Top 10 SubCategories')
    sns.countplot(data=df, x='baseColour', ax=axes[1,1], order=df['baseColour'].value_counts().index[:10])
    axes[1,1].set_title('Top 10 Base Colours')
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