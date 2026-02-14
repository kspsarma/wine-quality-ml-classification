# Wine Quality Classification System

## Problem Statement

Wine quality assessment is a critical process in the wine industry, traditionally performed by human experts through sensory analysis. However, this method is subjective, time-consuming, and expensive. The challenge is to develop an automated, objective system that can predict wine quality based on physicochemical properties measurable in laboratory tests.

This project addresses the following problems:
1. **Subjectivity in Quality Assessment**: Human tasters may have inconsistent evaluations
2. **Cost Efficiency**: Reducing the need for expensive expert sensory panels
3. **Scalability**: Enabling quality assessment for large production volumes
4. **Speed**: Providing rapid quality predictions during production

The objective is to build and compare multiple machine learning classification models that can accurately predict whether a wine is of "Good Quality" (≥6 rating) or "Average Quality" (<6 rating) based on 12 physicochemical attributes.

## Dataset Description

**Dataset Name**: Wine Quality Dataset

**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

**Original Citation**: 
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, Elsevier, 47(4):547-553, 2009.

### Dataset Characteristics

- **Total Instances**: 6,497 wine samples
  - Red Wine (Vinho Verde): 1,599 samples
  - White Wine (Vinho Verde): 4,898 samples
- **Number of Features**: 12 physicochemical properties + 1 derived feature
- **Classification Type**: Binary Classification
- **Target Variable**: 
  - Class 0: Average Wine (original quality < 6)
  - Class 1: Good Wine (original quality ≥ 6)
- **Class Distribution**:
  - Good Wine: 3,258 samples (50.2%)
  - Average Wine: 3,239 samples (49.8%)

### Feature Descriptions

| # | Feature Name | Description | Data Type | Unit | Range |
|---|--------------|-------------|-----------|------|-------|
| 1 | fixed acidity | Tartaric acid content - contributes to wine's fixed acidity | Continuous | g/dm³ | 3.8 - 15.9 |
| 2 | volatile acidity | Acetic acid content - high levels lead to unpleasant vinegar taste | Continuous | g/dm³ | 0.08 - 1.58 |
| 3 | citric acid | Adds freshness and flavor to wines | Continuous | g/dm³ | 0.0 - 1.66 |
| 4 | residual sugar | Sugar remaining after fermentation stops | Continuous | g/dm³ | 0.6 - 65.8 |
| 5 | chlorides | Salt content in wine | Continuous | g/dm³ | 0.009 - 0.611 |
| 6 | free sulfur dioxide | Prevents microbial growth and oxidation of wine | Continuous | mg/dm³ | 1 - 289 |
| 7 | total sulfur dioxide | Total amount of SO₂ (free + bound forms) | Continuous | mg/dm³ | 6 - 440 |
| 8 | density | Density of wine, depends on alcohol and sugar content | Continuous | g/cm³ | 0.987 - 1.039 |
| 9 | pH | Acidity level (most wines 3-4 on pH scale) | Continuous | pH scale | 2.72 - 4.01 |
| 10 | sulphates | Wine additive contributing to sulfur dioxide levels | Continuous | g/dm³ | 0.22 - 2.0 |
| 11 | alcohol | Alcohol percentage by volume | Continuous | % vol | 8.0 - 14.9 |
| 12 | wine_type | Type of wine (derived feature) | Binary | 0/1 | 0=White, 1=Red |

### Data Preprocessing

1. **Dataset Combination**: Merged red and white wine datasets
2. **Feature Engineering**: Added binary 'wine_type' indicator
3. **Target Transformation**: Converted multi-class quality (0-10) to binary classification
4. **Missing Values**: No missing values in the dataset
5. **Feature Scaling**: Applied StandardScaler normalization
6. **Train-Test Split**: 75% training, 25% testing with stratification

## Models Used

### Model Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|------|
| Logistic Regression | 0.7518 | 0.8255 | 0.7318 | 0.7808 | 0.7554 | 0.5032 |
| Decision Tree | 0.7394 | 0.7377 | 0.7235 | 0.7556 | 0.7392 | 0.4787 |
| kNN | 0.7240 | 0.7922 | 0.7032 | 0.7567 | 0.7289 | 0.4480 |
| Naive Bayes | 0.7271 | 0.8043 | 0.7023 | 0.7714 | 0.7353 | 0.4554 |
| Random Forest (Ensemble) | 0.7765 | 0.8548 | 0.7589 | 0.8023 | 0.7799 | 0.5531 |
| XGBoost (Ensemble) | 0.7734 | 0.8512 | 0.7543 | 0.8017 | 0.7773 | 0.5469 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Logistic Regression demonstrates solid baseline performance with 75.18% accuracy. The model achieves a strong AUC of 0.8255, indicating good discrimination ability between wine quality classes. With balanced precision (0.7318) and recall (0.7808), it shows reliable performance for both identifying good wines and avoiding false positives. The MCC score of 0.5032 indicates moderate correlation strength. This model benefits from the linear separability present in the wine chemistry features and provides the advantage of interpretability through feature coefficients. Training is fast and the model generalizes well without overfitting, making it suitable for production deployment where model transparency is valued. |
| Decision Tree | The Decision Tree achieves 73.94% accuracy with an AUC of 0.7377, showing moderate performance. While the model offers excellent interpretability through its rule-based structure, it demonstrates the lowest AUC among all models, suggesting limited ability to rank predictions by confidence. The precision (0.7235) and recall (0.7556) are reasonably balanced, but the MCC of 0.4787 indicates weaker correlation than ensemble methods. The tree depth was limited to 10 and minimum samples per split set to 20 to prevent overfitting, but this constraint appears to limit the model's ability to capture complex relationships in the wine chemistry data. The model would benefit from ensemble approaches to improve generalization. |
| kNN | k-Nearest Neighbors shows the weakest overall performance with 72.40% accuracy. Using k=7 neighbors with distance weighting and Euclidean metric, the model achieves an AUC of 0.7922, which is better than Decision Tree but falls short of other methods. The precision of 0.7032 indicates relatively high false positive rate, while recall of 0.7567 shows moderate sensitivity. The low MCC of 0.4480 suggests weak predictive correlation. This performance is likely due to the curse of dimensionality in the 12-dimensional feature space and sensitivity to feature scaling. The model's instance-based nature makes it memory-intensive and slow for predictions on large datasets, limiting its practical deployment value despite its simplicity. |
| Naive Bayes | Gaussian Naive Bayes achieves 72.71% accuracy with a respectable AUC of 0.8043, demonstrating good probabilistic ranking despite the conditional independence assumption. The model shows precision of 0.7023 and recall of 0.7714, with recall notably higher than precision, indicating a tendency to predict "Good Wine" more liberally. The MCC of 0.4554 suggests moderate correlation. While the independence assumption between features (e.g., alcohol content and density are actually correlated) is violated in wine chemistry, the model still performs reasonably well. Its extremely fast training and prediction times, along with minimal memory requirements, make it attractive for real-time applications. The probabilistic outputs are well-calibrated for threshold-based decision making. |
| Random Forest (Ensemble) | Random Forest delivers the best overall performance with 77.65% accuracy and the highest AUC of 0.8548, demonstrating superior ability to distinguish between quality classes. Using 150 trees with maximum depth of 15 and minimum 10 samples per split, the ensemble effectively captures complex non-linear relationships while avoiding overfitting. The model achieves the best precision (0.7589) and recall (0.8023) balance, with the highest F1 score (0.7799) and MCC (0.5531), indicating strong and reliable predictions. Feature importance analysis reveals that alcohol content, volatile acidity, and sulphates are key quality indicators. The model's robustness to outliers and ability to handle feature interactions make it the recommended choice for deployment despite higher computational costs. |
| XGBoost (Ensemble) | XGBoost achieves competitive performance with 77.34% accuracy and AUC of 0.8512, nearly matching Random Forest. The gradient boosting approach with 100 estimators and learning rate of 0.1 provides strong regularization, preventing overfitting while capturing complex patterns. With precision of 0.7543 and recall of 0.8017, the model shows excellent balance, and its MCC of 0.5469 confirms strong predictive power. XGBoost's built-in handling of missing values and feature importance calculations add practical value. While slightly underperforming Random Forest on this dataset, XGBoost offers faster training times and better scalability to larger datasets. The model would be the preferred choice when training efficiency is critical or when the dataset grows significantly larger. |

### Key Insights and Recommendations

**Performance Summary:**
- **Best Model**: Random Forest (77.65% accuracy, 0.8548 AUC)
- **Runner-up**: XGBoost (77.34% accuracy, 0.8512 AUC)
- **Weakest Model**: kNN (72.40% accuracy)
- **Best Recall**: Random Forest (0.8023) - Important for minimizing false negatives
- **Best Precision**: Random Forest (0.7589) - Minimizes false positives

**Deployment Recommendation**: For production use, **Random Forest** is recommended due to highest overall accuracy, best AUC score, robust performance, and excellent generalization. For real-time applications where speed is critical, **Logistic Regression** provides a good accuracy-speed tradeoff.

## Project Structure

```
wine-quality-classification/
│
├── streamlit_wine_app.py                          # Streamlit web application
├── train_wine_models.py                           # Model training pipeline
├── requirements.txt                               # Python dependencies
├── README.md                                      # Project documentation
│
├── wine_logistic_regression_classifier.pkl        # Trained models
├── wine_decision_tree_classifier.pkl
├── wine_knn_classifier.pkl
├── wine_naive_bayes_classifier.pkl
├── wine_random_forest_classifier.pkl
├── wine_xgboost_classifier.pkl
├── wine_quality_scaler.pkl                        # Feature scaler
│
├── wine_model_performance.csv                     # Performance metrics
└── wine_test_dataset.csv                          # Test data
```

## Installation & Usage

### Local Setup

```bash
# Clone repository
git clone <your-repo-url>
cd wine-quality-classification

# Install dependencies
pip install -r requirements.txt

# Train models
python train_wine_models.py

# Run Streamlit app
streamlit run streamlit_wine_app.py
```

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" → Select repository
5. Choose branch (main) and main file (streamlit_wine_app.py)
6. Click "Deploy"

## Streamlit App Features

✅ **Dataset Upload**   : CSV upload with validation  
✅ **Model Selection**  : Dropdown with all 6 models  
✅ **Metrics Display**  : All 6 evaluation metrics  
✅ **Confusion Matrix** : Interactive heatmap + report  

**Additional Features**: Interactive charts, comparative analysis, model rankings, responsive design

## Technologies Used

- Python 3.8+
- Streamlit 1.31.0
- Scikit-learn 1.4.0
- XGBoost 2.0.3
- Pandas, NumPy, Plotly

---