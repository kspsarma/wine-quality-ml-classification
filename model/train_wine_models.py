"""
Wine Quality Classification System
Author: SHANKARA PHANINDRASARMA KAVIRAYANI.
Date: 10th February 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("WINE QUALITY CLASSIFICATION - MODEL TRAINING PIPELINE")
print("="*70)

# Dataset Loading and Preparation
print("\n[STEP 1] Loading Wine Quality Dataset from UCI Repository...")
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Load both red and white wine datasets
red_df = pd.read_csv(red_wine_url, sep=';')
white_df = pd.read_csv(white_wine_url, sep=';')

# Add wine type indicator
red_df['wine_type'] = 1  # Red wine
white_df['wine_type'] = 0  # White wine

# Combine datasets
combined_wine_data = pd.concat([red_df, white_df], axis=0, ignore_index=True)

print(f"✓ Dataset loaded successfully")
print(f"  - Total samples: {combined_wine_data.shape[0]}")
print(f"  - Total features: {combined_wine_data.shape[1] - 1}")
print(f"  - Red wine samples: {len(red_df)}")
print(f"  - White wine samples: {len(white_df)}")

# Data Preprocessing
print("\n[STEP 2] Preprocessing and Feature Engineering...")

# Create binary classification target
# Quality >= 6 is considered "Good Wine" (1), otherwise "Average Wine" (0)
combined_wine_data['quality_class'] = (combined_wine_data['quality'] >= 6).astype(int)

# Remove original quality score (keep only binary class)
wine_features = combined_wine_data.drop(['quality', 'quality_class'], axis=1)
wine_target = combined_wine_data['quality_class']

print(f"✓ Binary classification created:")
print(f"  - Good Wine (quality >= 6): {wine_target.sum()} samples")
print(f"  - Average Wine (quality < 6): {(wine_target == 0).sum()} samples")

# Feature Information
feature_names = wine_features.columns.tolist()
print(f"\n✓ Feature set ({len(feature_names)} features):")
for idx, feat in enumerate(feature_names, 1):
    print(f"  {idx}. {feat}")

# Train-Test Split with stratification
print("\n[STEP 3] Splitting dataset into train/test sets...")
X_train_data, X_test_data, y_train_labels, y_test_labels = train_test_split(
    wine_features, wine_target, 
    test_size=0.25, 
    random_state=42, 
    stratify=wine_target
)

print(f"✓ Split completed:")
print(f"  - Training samples: {X_train_data.shape[0]}")
print(f"  - Testing samples: {X_test_data.shape[0]}")

# Feature Scaling
print("\n[STEP 4] Applying feature standardization...")
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train_data)
X_test_scaled = feature_scaler.transform(X_test_data)

# Save scaler for deployment
with open('wine_quality_scaler.pkl', 'wb') as scaler_file:
    pickle.dump(feature_scaler, scaler_file)
print("✓ Scaler saved: wine_quality_scaler.pkl")

# Save test data for Streamlit app
test_dataset = pd.DataFrame(X_test_scaled, columns=feature_names)
test_dataset['quality_class'] = y_test_labels.values
test_dataset.to_csv('wine_test_dataset.csv', index=False)
print("✓ Test dataset saved: wine_test_dataset.csv")

# Model Configuration
print("\n[STEP 5] Initializing classification models...")
classification_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, solver='lbfgs'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20),
    'kNN': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean'),
    'Naive Bayes': GaussianNB(var_smoothing=1e-9),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, max_depth=15, min_samples_split=10),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100, learning_rate=0.1)
}

print(f"✓ {len(classification_models)} models configured")

# Model Training and Evaluation
performance_results = []

print("\n" + "="*70)
print("MODEL TRAINING AND EVALUATION")
print("="*70)

for model_identifier, classifier_obj in classification_models.items():
    print(f"\n{'─'*70}")
    print(f"► Training: {model_identifier}")
    print(f"{'─'*70}")
    
    # Train the classifier
    classifier_obj.fit(X_train_scaled, y_train_labels)
    
    # Generate predictions
    predictions = classifier_obj.predict(X_test_scaled)
    
    # Get probability scores for AUC
    if hasattr(classifier_obj, 'predict_proba'):
        probability_scores = classifier_obj.predict_proba(X_test_scaled)[:, 1]
    else:
        probability_scores = predictions
    
    # Calculate all evaluation metrics
    acc_score = accuracy_score(y_test_labels, predictions)
    auc_score = roc_auc_score(y_test_labels, probability_scores)
    prec_score = precision_score(y_test_labels, predictions, average='binary', zero_division=0)
    rec_score = recall_score(y_test_labels, predictions, average='binary', zero_division=0)
    f1_metric = f1_score(y_test_labels, predictions, average='binary', zero_division=0)
    mcc_metric = matthews_corrcoef(y_test_labels, predictions)
    
    # Display results
    print(f"  Accuracy Score:               {acc_score:.4f}")
    print(f"  AUC (ROC) Score:              {auc_score:.4f}")
    print(f"  Precision Score:              {prec_score:.4f}")
    print(f"  Recall Score:                 {rec_score:.4f}")
    print(f"  F1-Score:                     {f1_metric:.4f}")
    print(f"  Matthews Correlation Coeff:   {mcc_metric:.4f}")
    
    # Store results
    performance_results.append({
        'Model': model_identifier,
        'Accuracy': round(acc_score, 4),
        'AUC': round(auc_score, 4),
        'Precision': round(prec_score, 4),
        'Recall': round(rec_score, 4),
        'F1': round(f1_metric, 4),
        'MCC': round(mcc_metric, 4)
    })
    
    # Save trained model
    model_filename = f"wine_{model_identifier.replace(' ', '_').lower()}_classifier.pkl"
    with open(model_filename, 'wb') as model_file:
        pickle.dump(classifier_obj, model_file)
    print(f"  ✓ Model saved: {model_filename}")

# Create results comparison DataFrame
results_comparison = pd.DataFrame(performance_results)
results_comparison.to_csv('wine_model_performance.csv', index=False)

print("\n" + "="*70)
print("FINAL PERFORMANCE COMPARISON")
print("="*70)
print(results_comparison.to_string(index=False))

print("\n" + "="*70)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n📁 Generated Files:")
print("  ✓ wine_quality_scaler.pkl")
print("  ✓ wine_test_dataset.csv")
print("  ✓ wine_model_performance.csv")
print("  ✓ 6 classifier model files (.pkl)")
print("\n✅ All models trained and saved successfully!")
print("="*70)
