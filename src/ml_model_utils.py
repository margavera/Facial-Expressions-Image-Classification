from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_features(img_paths, model, datagen):
    features = []
    for path in img_paths:
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        # Apply data augmentation
        img_array = datagen.random_transform(img_array)  # Apply a random transformation
        features.append(model.predict(np.expand_dims(img_array, axis=0))[0])
    return np.array(features).reshape(len(img_paths), -1)


def train_and_evaluate_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains an SVM classifier and evaluates it on validation and test sets.

    Parameters:
    - X_train, y_train: training features and labels
    - X_val, y_val: validation features and labels
    - X_test, y_test: test features and labels

    Returns:
    - clf: trained SVM model
    """
    clf = SVC()
    clf.fit(X_train, y_train)

    print("Validation Classification Report:")
    y_pred_val = clf.predict(X_val)
    print(classification_report(y_val, y_pred_val))

    print("Test Classification Report:")
    y_pred_test = clf.predict(X_test)
    print(classification_report(y_test, y_pred_test))

    return clf

def train_and_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains a Random Forest classifier and evaluates it on validation and test sets.

    Parameters:
    - X_train, y_train: training features and labels
    - X_val, y_val: validation features and labels
    - X_test, y_test: test features and labels

    Returns:
    - clf_rf: trained Random Forest model
    """
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)

    print("Validation Classification Report:")
    y_pred_val = clf_rf.predict(X_val)
    print(classification_report(y_val, y_pred_val))

    print("Test Classification Report:")
    y_pred_test = clf_rf.predict(X_test)
    print(classification_report(y_test, y_pred_test))

    return clf_rf

def plot_confusion_matrix(y_true, y_pred, classes=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix heatmap.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - classes: list of class names for axis labels (optional)
    - title: title of the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes if classes is not None else "auto",
        yticklabels=classes if classes is not None else "auto"
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def evaluate_classification_metrics(y_true, y_pred, classes=None, title="Confusion Matrix"):
    """
    Computes and displays classification metrics and a confusion matrix.

    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - classes: list of class names for the confusion matrix (optional)
    - title: title for the confusion matrix plot

    Returns:
    - metrics_df: classification metrics as a pandas DataFrame
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report_dict).transpose()

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", metrics_df)

    plot_confusion_matrix(y_true, y_pred, classes=classes, title=title)

    return metrics_df
