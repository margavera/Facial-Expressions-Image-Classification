import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def convert_grayscale_to_3channel(img):
    """
    Convert grayscale images to 3-channel (duplicating the grayscale channel).
    This function is used within the data generator preprocessing.
    """
    # Convert to grayscale first if it's RGB
    if len(img.shape) == 3 and img.shape[-1] == 3:
        # Already 3 channels, convert to grayscale then back to 3 channels
        gray = tf.image.rgb_to_grayscale(img)
        return tf.tile(gray, [1, 1, 3])
    elif len(img.shape) == 3 and img.shape[-1] == 1:
        # Already grayscale, just duplicate to 3 channels
        return tf.tile(img, [1, 1, 3])
    else:
        # Reshape single channel to 3 channels (for PIL images)
        img = tf.expand_dims(img, -1)
        return tf.tile(img, [1, 1, 3])

def get_data_generators(
    train_df,
    img_size=(224, 224),
    batch_size=16,
    val_split=0.2,
    seed=42,
    x_col='image_path',
    y_col='Regruped Emotion',
    color_mode='rgb'
):
    """
    Returns training and validation data generators from a DataFrame.
    Uses stratified splitting to ensure balanced class distribution.
    
    Args:
        train_df: DataFrame with image paths and labels
        img_size: Target image size
        batch_size: Batch size for training
        val_split: Validation split ratio
        seed: Random seed for reproducibility
        x_col: Column name for image paths
        y_col: Column name for class labels
        color_mode: Initial color mode for loading (default 'rgb')
    """
    
    # Stratified split to ensure class balance between train and validation
    train_df, val_df = train_test_split(
        train_df, 
        test_size=val_split, 
        stratify=train_df[y_col],  # Ensure the split is stratified by the target class
        random_state=seed
    )
    
    # Data augmentation for training
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        preprocessing_function=convert_grayscale_to_3channel  # Convert to 3-channel
    )
    
    # Training data generator
    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col=x_col,
        y_col=y_col,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=True,
        seed=seed
    )
    
    # Validation data generator
    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col=x_col,
        y_col=y_col,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False
    )
    
    return train_gen, val_gen

def get_test_generator(
    test_df,
    img_size=(224, 224),
    batch_size=16,
    x_col='image_path',
    y_col='Regruped Emotion',
    color_mode='rgb' 
):
    """
    Returns a test data generator from a DataFrame.
    Uses grayscale images but converts them to 3-channel format for ResNet50.
    """
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=convert_grayscale_to_3channel  # Convert to 3-channel
    )
    
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col=x_col,
        y_col=y_col,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,  
        shuffle=False
    )
    
    return test_gen

def build_resnet50(input_shape, num_classes, freeze_layers=True):
    """
    Build a model using pre-trained ResNet50 for facial expression recognition.
    
    Args:
        input_shape: Input image shape (must be (height, width, 3) for ResNet50)
        num_classes: Number of output classes (emotion categories)
        freeze_layers: Whether to freeze the pre-trained layers
        
    Returns:
        Compiled ResNet50 model
    """
    # Make sure input shape has 3 channels
    if input_shape[2] != 3:
        input_shape = (input_shape[0], input_shape[1], 3)
        
    # Create the base pre-trained model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model if requested
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_resnet101(input_shape, num_classes, freeze_layers=True):
    """
    Build a model using pre-trained ResNet101 for facial expression recognition.
    
    Args:
        input_shape: Input image shape (must be (height, width, 3) for ResNet101)
        num_classes: Number of output classes (emotion categories)
        freeze_layers: Whether to freeze the pre-trained layers
        
    Returns:
        Compiled ResNet101 model
    """
    # Make sure input shape has 3 channels
    if input_shape[2] != 3:
        input_shape = (input_shape[0], input_shape[1], 3)
        
    # Create the base pre-trained model
    base_model = ResNet101(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model if requested
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Lower learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_gen, val_gen, epochs=20, out_path='resnet50_best.keras'):
    """
    Train the model with callbacks and return the history.
    """
    callbacks = [
        # Early stopping
        EarlyStopping(
            patience=10,
            restore_best_weights=True, 
            monitor='val_loss',
            verbose=1
        ),
        # Save best model
        ModelCheckpoint(
            out_path, 
            save_best_only=True, 
            monitor='val_loss',
            verbose=1
        ),
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def unfreeze_and_finetune(model, train_gen, val_gen, n_layers=2, epochs=10, out_path='resnet50_finetuned.keras'):
    """
    Unfreeze the last 'n_layers' layers of the ResNet model and fine-tune the entire model with a very low learning rate.
    
    Args:
        model: Trained model with frozen layers
        train_gen: Training data generator
        val_gen: Validation data generator
        n_layers: Number of layers to unfreeze from the end of the model
        epochs: Number of fine-tuning epochs
        out_path: Path to save the fine-tuned model
        
    Returns:
        Training history
    """
    # Unfreeze the last 'n_layers' layers
    for layer in model.layers[-n_layers:]:  # Unfreeze the last 'n_layers' layers
        layer.trainable = True
    
    # Recompile the model with a very low learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Very low learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks for fine-tuning
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss', verbose=1),
        ModelCheckpoint(out_path, save_best_only=True, monitor='val_loss', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    ]
    
    # Fine-tune the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def display_training_curves(history):
    """
    Display training and validation accuracy/loss curves
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, test_gen, class_indices, num_samples=10):
    """
    Visualize model predictions on test data.
    
    Args:
        model: Trained model
        test_gen: Test data generator
        class_indices: Dictionary mapping class indices to class names
        num_samples: Number of samples to visualize
        
    Returns:
        None (displays images with predictions)
    """
    # Get class names
    class_names = {v: k for k, v in class_indices.items()}
    
    # Get a batch of test data
    test_gen.reset()
    batch_x, batch_y = next(test_gen)
    
    # Make predictions
    predictions = model.predict(batch_x)
    
    # Plot images with predictions
    plt.figure(figsize=(20, num_samples * 2))
    
    for i in range(min(num_samples, len(batch_x))):
        plt.subplot(num_samples//5 + 1, 5, i+1)
        
        # Display grayscale image (first channel since all 3 channels are the same)
        plt.imshow(batch_x[i][:,:,0], cmap='gray')
        plt.axis('off')
        
        # Get true and predicted classes
        true_class = np.argmax(batch_y[i])
        pred_class = np.argmax(predictions[i])
        
        # Get class names
        true_class_name = class_names[true_class]
        pred_class_name = class_names[pred_class]
        
        # Set title color (green for correct, red for incorrect)
        title_color = 'green' if true_class == pred_class else 'red'
        
        # Display prediction
        plt.title(f"True: {true_class_name}\nPred: {pred_class_name}", 
                  color=title_color, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_gen, class_names=None, num_batches=None, figsize=(10, 8)):
    """
    Evaluate a trained model on a test data generator, creating a confusion matrix 
    and classification report.
    
    Args:
        model: Trained Keras model
        test_gen: Test data generator (e.g., from flow_from_dataframe)
        class_names: List of class names (if None, will use test_gen.class_indices)
        num_batches: Number of batches to evaluate (if None, use len(test_gen))
        figsize: Size of the confusion matrix figure (tuple)
        
    Returns:
        dict: Dictionary containing evaluation metrics
        
    Example:
        metrics = evaluate_model(final_model, test_gen)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    # Reset the generator to ensure we start from the beginning
    test_gen.reset()
    
    # If class_names not provided, use the class indices from test_gen
    if class_names is None:
        class_names = list(test_gen.class_indices.keys())
    
    # If num_batches not provided, use all batches
    if num_batches is None:
        num_batches = len(test_gen)
    
    # Collect true and predicted labels
    y_true = []
    y_pred = []
    
    # Iterate through batches
    for i in range(num_batches):
        batch_x, batch_y = next(test_gen)
        batch_pred = model.predict(batch_x)
        
        # Convert from one-hot encoding to class indices
        batch_y_indices = np.argmax(batch_y, axis=1)
        batch_pred_indices = np.argmax(batch_pred, axis=1)
        
        y_true.extend(batch_y_indices)
        y_pred.extend(batch_pred_indices)
    
    # Convert to integer arrays
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calculate individual metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    return metrics

def evaluate_model_per_class(metrics, class_names):
    """
    Generate per-class evaluation metrics from the output of evaluate_model.
    
    Args:
        metrics: Dictionary from evaluate_model function
        class_names: List of class names
        
    Returns:
        pd.DataFrame: DataFrame with per-class precision, recall, and F1-score
    """
    # Get per-class metrics
    precision = precision_score(metrics['y_true'], metrics['y_pred'], average=None)
    recall = recall_score(metrics['y_true'], metrics['y_pred'], average=None)
    f1 = f1_score(metrics['y_true'], metrics['y_pred'], average=None)
    
    # Create a DataFrame for visualization
    class_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Plot per-class metrics
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='value', hue='metric', 
                data=pd.melt(class_metrics, id_vars=['Class'], value_vars=['Precision', 'Recall', 'F1-Score'],
                            var_name='metric', value_name='value'))
    plt.title('Per-Class Performance Metrics')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return class_metrics

