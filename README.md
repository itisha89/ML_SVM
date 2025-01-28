# README: Support Vector Machines for Ad-Click Prediction and Wine-Type Prediction

## Project Overview
This project involves the implementation of Support Vector Machines (SVMs) for two classification tasks:

1. **Ad-Click Prediction**
   - **Source**: Kaggle
   - **Dataset Size**: 400 observations, 5 features
   - **Target Classes**: 2 (clicked, not_clicked)

2. **Wine-Type Prediction**
   - **Source**: UCI Machine Learning Repository
   - **Dataset Size**: 178 observations, 17 features
   - **Target Classes**: 3 (wine types)

The objective is to explore and compare SVM performance on binary and multi-class classification problems, using techniques such as feature standardization, PCA, class imbalance handling, and hyperparameter tuning.

---

## Data Preparation

### Exploratory Data Analysis (EDA) and Cleaning
- Both datasets were clean with **no NULL values**.
- **Categorical features** were encoded for compatibility with the SVM model.

### Handling Data Imbalance (Ad-Click Dataset)
- The `not_clicked` class was twice as frequent as the `clicked` class.
- **Class Weights Method**: Used `sklearn’s compute_class_weight` to assign weights, balancing the class distribution.
- Compared with alternative imbalance handling techniques (RandomUnderSampler, SMOTE).

### Baseline Accuracy
- A **DummyClassifier** predicting the most frequent class was trained for baseline accuracy.
- **Baseline Accuracy**: 64.2%
- SVM models aimed to outperform this baseline.

---

## Machine Learning Pipeline
1. **Feature Standardization**
   - Standardized the features for optimal SVM performance.

2. **Principal Component Analysis (PCA)**
   - Added PCA to reduce dimensionality and improve model efficiency.

3. **SVM Training**
   - Models were trained using different kernels: RBF, polynomial, and linear.
   - **Hyperparameter Tuning**: GridSearchCV with StratifiedKFold was used to optimize hyperparameters such as `C`, `gamma`, and PCA components.

4. **Model Evaluation**
   - The RBF kernel achieved the best performance:
     - **Test Set Accuracy**: 90%
     - **Full Dataset Accuracy**: 91%
   - Performance metrics included precision, recall, F1-score, and accuracy.
   - Nested cross-validation provided an unbiased estimate of model accuracy: **Mean Score**: 0.900 ± 0.018

---

## Results: Ad-Click Prediction

### Class Imbalance Handling Techniques
- **Class Weights Approach**: Balanced performance, effective handling of imbalance.
- **RandomUnderSampler**: Comparable performance to class weights.
- **SMOTE**: Slightly lower performance than class weights and RandomUnderSampler.
- **No Approach**: Lower performance than imbalance-handling techniques.

### Kernel Comparisons
- **RBF Kernel**: Highest accuracy (Test: 90%, Full Dataset: 91%).
- **Linear Kernel**: Moderate accuracy (Test: 88%, Full Dataset: 85%).
- **Polynomial Kernel**: Lowest test accuracy (82%) but matched RBF on the full dataset (91%).

### Conclusion
The **RBF kernel** consistently outperformed linear and polynomial kernels for the ad-click dataset.

---

## Results: Wine-Type Prediction

### Model Evaluation
- **Best Model**: Obtained from GridSearchCV using the RBF kernel.
- **Strategies for Multi-Class Classification**:
  1. **One-vs-All (OvA)**: Simpler and more effective.
  2. **Error Correcting Output Codes (ECOC)**: Performed slightly worse due to the complexity of its coding matrix.

### Key Observations
- OvA strategy achieved superior performance due to better hyperparameter optimization and efficient training.
- ECOC’s complexity made hyperparameter tuning less effective, resulting in lower accuracy.

---

## Summary of Findings

1. **Ad-Click Prediction**:
   - RBF kernel achieved the highest performance (Test Accuracy: 90%, Full Dataset: 91%).
   - Class imbalance handling techniques (Class Weights, RandomUnderSampler) significantly improved model performance.

2. **Wine-Type Prediction**:
   - OvA strategy outperformed ECOC for multi-class classification.
   - RBF kernel was the best-performing model across different evaluation strategies.

---

## Project Highlights
- **Baseline Accuracy**: 64.2%
- **Final Model Accuracy (Ad-Click Prediction)**: 91%
- **Cross-Validation Mean Score**: 0.900 ± 0.018
- Effective handling of class imbalance and dimensionality reduction using class weights and PCA.
- RBF kernel’s superior performance validated across binary and multi-class tasks.

---

## Future Work
- Explore ensemble methods like bagging or boosting to enhance SVM performance.
- Investigate additional imbalance-handling techniques for improved robustness.
- Apply SVMs to larger, more complex datasets to evaluate scalability and generalizability.

---

## References
- Kaggle (Ad-Click Dataset)
- UCI Machine Learning Repository (Wine Dataset)
- Scikit-learn Documentation for SVM, GridSearchCV, and Class Weights

