# Facial-Expressions-Image-Classification
Team members:

| Name                           | Email                               |
| -----------------------        | ----------------------------------- |
| Margarita Vera Cabrer          | marga.vera@alu.icai.comillas.edu    |
| Elena Martínez Torrijos        | 202407060@alu.comillas.edu          |
| Claudia Hermández de la Calera | chdelacalera@alu.comillas.edu       |

### Dataset:
The dataset used in this project was obtained from Kaggle and is available at the following link:     
[Facial Expression Recognition](https://www.kaggle.com/datasets/msambare/fer2013/data)

The objective is to classify the facial expressions into one of seven emotion categories: 
- 0 = Angry
- 1 = Disgust
- 2 = Fear
- 3 = Happy
- 4 = Sad
- 5 = Surprise
- 6 = Neutral

The dataset is split into a training set with 28,709 images and a public test set with 3,589 images.

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
│   ├── dl_model_utils.py                    # Deep Learning utils
│   └── eda_utils.py                         # EDA utils
│   └── ml_model_utils.py                    # Machine Learning utils      
├── README.md                                # This file
└── requirements.txt                       # Evinoment's requirements
```

**Model Folder**

The trained models and model artifacts are stored in the **`models/`** directory. However, due to their size, they could not be uploaded directly to the repository. You can access the models via the following Google Drive link:

[**Models on Google Drive**](https://drive.google.com/drive/folders/1TTkPwU9tVEYAOiKH9W_90rX-rAiAYPez?usp=sharing)

### Project Overview:
In this project, we aim to classify facial expressions into three categories: **positive**, **negative**, and **neutral**. Initially, we planned to use all the images with their corresponding categories. However, due to dataset size and resource limitations, we reduced the number of images and regrouped the categories into three: `positive`, `negative`, and `neutral`. This helped manage the computational load and focus on the most relevant categories for our task.

### Lines of Improvement:

1. **Image Detection for Preprocessing**:
   It would be beneficial to use **face detection** techniques as part of the preprocessing pipeline. This would help focus the model’s attention on the faces, eliminating noise from the background and improving the overall model performance. By centering the face in the image and normalizing it, the model would have cleaner data to train on, leading to better results.

2. **Better Preprocessing for Machine Learning Models**:
   For the **Machine Learning models**, better preprocessing could improve accuracy. Specifically, **SVM** has yielded the best accuracy among all trained models, possibly because these models use **feature extraction** techniques that first convert the images into vectors. If the quality of the images was poor, the feature extraction process could still capture the relevant features, making the model more robust. Future work should focus on improving image preprocessing and data augmentation techniques.

3. **Deep Learning Models (ResNet50 and ResNet101)**:
   Both the **ResNet50** and **ResNet101** models, when fine-tuned by unfreezing layers, yielded an accuracy of approximately **35%**. Both models show similar performance, with **ResNet50** predominantly classifying most of the predictions as `positive`, and **ResNet101** focusing on `neutral`. Despite the classes being well-balanced, these results are likely due to insufficient computational resources.

   - **Current Challenge**: We do not have access to a powerful GPU, which limits our ability to experiment extensively with different configurations to find the optimal setup for the model. Given this limitation, we have not been able to explore fine-tuning further or try more complex architectures that could potentially improve performance.

4. **Improving Performance**:
   - **Further Layer Unfreezing**: More layers could be unfrozen to allow the models to learn more fine-grained features from the data.
   - **Increased Computational Resources**: Using more computational power, such as access to high-performance GPUs, would likely allow for deeper experimentation and improved results, especially for **ResNet50** and **ResNet101**.

5. **Future Directions**:
   - Experiment with different architectures, like **VGG16**, or use **transfer learning** with additional pre-trained models.
   - Apply more advanced **data augmentation** and **regularization techniques** to reduce overfitting and improve generalization on unseen data.
   - Investigate **multi-task learning** for better integration of related tasks such as emotion classification alongside other facial feature recognition tasks.

### Conclusion:
This project serves as a preliminary attempt to classify facial expressions into a simplified set of categories using **Deep Learning** and **Machine Learning** models. While the results achieved so far are promising, especially with **SVM**, there is significant room for improvement with better preprocessing, access to more computational resources, and fine-tuning of deeper models.

We are hopeful that with further experimentation, especially with more advanced techniques like **face detection** and **deep neural networks**, we can significantly improve the accuracy and robustness of our models.
