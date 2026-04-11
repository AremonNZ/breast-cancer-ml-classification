Project Overview: This project builds a machine learning model to classify breast cancer tumors as malignant or bengin using the Wisconsin Breast Cancer dataset.
The goal is to evaluate different models and identify the most effective approach for accurate diagnosis.
Dataset: 569 samples, 30 numerical features
Target variable: 0 = Benign , 1 = Malignant , No missing values , Slight class imbalance
Data Preprocessing: Label encoding applied to target variable , Feature scalling using StandardScaler , Stratified train-test split (70/30)
Model Used : Logistic Regression (baseline model) , Multi-Layer Perceptron ( MLP neural network)
Results :

- MLP Accuracy (Test Set):97.66%  
- Training Accuracy:98.74%  
- Test Accuracy:97.66%  

The model demonstrates strong generalisation with minimal overfitting, as indicated by similar training and test performance.

 Classification Performance:
- High precision and recall for both classes
- Slightly lower recall for malignant cases, indicating some false negatives

Interpretation:

- The model performs strongly overall, achieving high accuracy and balanced precision/recall.
- However, in a medical context, **false negatives (missing malignant cases)** are critical.
- This suggests that further optimisation should prioritise recall for the malignant class rather than overall accuracy.

- Logistic Regression Accuracy: 0.98%
- MLP Accuracy: 97.66%

The MLP model outperformed the baseline Logistic Regression model, indicating its ability to capture more complex, non-linear relationships in the data.
