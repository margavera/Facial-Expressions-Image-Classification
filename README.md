# Facial-Expressions-Image-Classification
Team members:

| Name                           | Email                               |
| -----------------------        | ----------------------------------- |
| Margarita Vera Cabrer          | marga.vera@alu.icai.comillas.edu    |
| Elena Martínez Torrijos        | 202407060@alu.comillas.edu          |
| Claudia Hermández de la Calera | chdelacalera@alu.comillas.edu       |


The dataset used in this project was obtained from Kaggle and is available at the following link:     
[Facial Expression Recognition](https://www.kaggle.com/datasets/msambare/fer2013/data)


The dataset contains 48x48 pixel grayscale images of human faces. Each face has been preprocessed to ensure it's roughly centered and uniformly scaled across all samples. The objective is to classify the facial expressions into one of seven emotion categories: 0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy, 4 = Sad, 5 = Surprise, and 6 = Neutral. The dataset is split into a training set with 28,709 images and a public test set with 3,589 images.

## Project Structure

```
.
├── data/                                    # Dataset files
├── models/                                  # Trained models and model artifacts
│   ├── resnet50_best.keras                  # DL Resnet 50 model
│   ├── resnet50_finetuned.keras             # DL Resnet 50 model, 1 frozen layer
│   ├── resnet50_finetuned2.keras            # DL Resnet 50 model, 2 frozen layers
│   ├── resnet101_best.keras                 # DL Resnet 101 model
│   ├── resnet101_finetuned.keras            # DL Resnet 101 model, 1 frozen layer
├── notebooks/                               # Jupyter notebooks
│   ├── EDA_Facial_Expressions.ipynb         # Exploratory Data Analysis
│   ├── ML_simple.ipynb                      # Random Forest and SVM models training and evaluation
│   ├── Resnet50.ipynb                       # DL model training and evaluation
│   └── Resnet101.ipynb                      # DL model training and evaluation
├── src/                                     # Source code
│   ├── 
│   └──                             
├── README.md                                # This file
└── requirements.txt                       # Evinoment's requirements
```
