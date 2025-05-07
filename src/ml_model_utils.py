from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PIL import Image
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def extract_features(img_paths, vgg_model):
    features = []
    for path in img_paths:
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        # Expand dims to match model input and predict
        feature = vgg_model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
        features.append(feature)
    # Flatten the features
    return np.array(features).reshape(len(img_paths), -1)

def train_and_evaluate_svm(X, y, test_size=0.2, random_state=42):
    """
    Trains an SVM classifier and evaluates its performance using classification_report.

    Parameters:
    - X: extracted features (array or DataFrame)
    - y: labels (Series or array)
    - test_size: proportion of the validation set
    - random_state: for reproducibility

    Returns:
    - clf: trained SVM model
    - y_pred: predictions on the validation set
    - y_val: ground truths
    """
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # Train the model
    clf = SVC()
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_val)
    print("Classification Report:\n")
    print(classification_report(y_val, y_pred))

    return clf, y_pred, y_val

def evaluate_classification_metrics(y_true, y_pred):
    """
    Computes and displays classification metrics and a confusion matrix.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels

    Returns:
    - metrics_df: classification metrics as a DataFrame
    """
    # Generate metrics
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report_dict).transpose()

    # Print accuracy
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", metrics_df)

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return metrics_df
