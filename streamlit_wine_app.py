"""
Wine Quality Prediction Dashboard
Interactive ML Classification System
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for unique styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #722F37;
        text-align: center;
        font-weight: bold;
        padding: 20px;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #722F37;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown('<div class="main-header">🍷 Wine Quality Classification System</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    st.markdown("""
    <div style='text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
        <p style='margin: 0; color: #495057;'><b>Machine Learning Assignment 2</b></p>
        <p style='margin: 5px 0; color: #6c757d; font-size: 0.9rem;'>M.Tech (AIML) | BITS Pilani</p>
        <p style='margin: 0; color: #6c757d; font-size: 0.85rem;'>Multi-Model Wine Classification Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# MODEL MAPPING
# ============================================================================
CLASSIFIER_MAPPING = {
    'Logistic Regression': 'model/wine_logistic_regression_classifier.pkl',
    'Decision Tree': 'model/wine_decision_tree_classifier.pkl',
    'kNN': 'model/wine_knn_classifier.pkl',
    'Naive Bayes': 'model/wine_naive_bayes_classifier.pkl',
    'Random Forest': 'model/wine_random_forest_classifier.pkl',
    'XGBoost': 'model/wine_xgboost_classifier.pkl'
}

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/wine.png", width=80)
    st.title("⚙️ Configuration Panel")
    
    st.markdown("---")
    
    st.subheader("🤖 Select ML Model")
    selected_classifier = st.selectbox(
        "Choose a classification algorithm:",
        options=list(CLASSIFIER_MAPPING.keys()),
        help="Select one of the 6 trained models for wine quality prediction",
        key="model_selector"
    )
    
    st.markdown("---")
    
    st.subheader("📊 Dataset Information")
    st.info("""
    **Wine Quality Dataset**
    - **Source**: UCI ML Repository
    - **Samples**: 6,497 wines
    - **Features**: 12 attributes
    - **Target**: Binary Classification
      - Good Wine (quality ≥ 6)
      - Average Wine (quality < 6)
    - **Types**: Red & White wines
    """)
    
    st.markdown("---")
    
    st.subheader("🎯 Model Details")
    model_info = {
        'Logistic Regression': 'Linear classification with regularization',
        'Decision Tree': 'Rule-based hierarchical splitting',
        'kNN': 'Distance-based similarity matching',
        'Naive Bayes': 'Probabilistic Bayesian classifier',
        'Random Forest': 'Ensemble of decision trees',
        'XGBoost': 'Gradient boosted trees'
    }
    st.markdown(f"**{selected_classifier}**")
    st.caption(model_info[selected_classifier])

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
@st.cache_resource
def load_classifier_model(model_name):
    """Load the selected classifier from disk"""
    try:
        model_path = CLASSIFIER_MAPPING[model_name]
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"⚠️ Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_feature_scaler():
    """Load the feature scaler"""
    try:
        with open('model/wine_quality_scaler.pkl', 'rb') as file:
            return pickle.load(file)
    except:
        st.warning("⚠️ Scaler file not found. Proceeding without scaling.")
        return None

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================
tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📈 Model Performance", "ℹ️ About Dataset"])

# TAB 1: UPLOAD AND PREDICT
with tab1:
    st.subheader("📁 Upload Wine Test Dataset")
    
    uploaded_csv = st.file_uploader(
        "Upload CSV file with wine chemical properties",
        type=['csv'],
        help="Upload the test dataset CSV file for wine quality prediction",
        key="csv_uploader"
    )
    
    if uploaded_csv is not None:
        try:
            # Load dataset
            wine_test_df = pd.read_csv(uploaded_csv)
            
            st.success(f"✅ Dataset uploaded successfully! Shape: {wine_test_df.shape}")
            
            # Check for target column
            has_labels = 'quality_class' in wine_test_df.columns
            
            if has_labels:
                X_features = wine_test_df.drop('quality_class', axis=1)
                y_actual = wine_test_df['quality_class']
            else:
                X_features = wine_test_df
                y_actual = None
                st.info("ℹ️ No 'quality_class' column found. Predictions only mode.")
            
            # Data Preview Section
            with st.expander("🔍 View Dataset Preview", expanded=False):
                st.dataframe(wine_test_df.head(15), use_container_width=True)
                
                preview_col1, preview_col2, preview_col3, preview_col4 = st.columns(4)
                preview_col1.metric("Total Samples", wine_test_df.shape[0])
                preview_col2.metric("Features", X_features.shape[1])
                if has_labels:
                    preview_col3.metric("Good Wines", int(y_actual.sum()))
                    preview_col4.metric("Average Wines", int((y_actual == 0).sum()))
            
            st.markdown("---")
            
            # Load model and scaler
            classifier = load_classifier_model(selected_classifier)
            scaler = load_feature_scaler()
            
            if classifier is not None:
                st.subheader(f"🤖 Predictions: {selected_classifier}")
                
                # Scale features
                if scaler is not None:
                    X_processed = scaler.transform(X_features)
                else:
                    X_processed = X_features.values
                
                # Make predictions
                with st.spinner("🔄 Generating predictions..."):
                    y_predictions = classifier.predict(X_processed)
                    
                    if hasattr(classifier, 'predict_proba'):
                        y_probabilities = classifier.predict_proba(X_processed)[:, 1]
                    else:
                        y_probabilities = y_predictions
                
                # CASE 1: WITH LABELS (EVALUATION MODE)
                if has_labels:
                    st.success("✅ Predictions complete! Evaluation metrics calculated.")
                    
                    # Calculate metrics
                    acc = accuracy_score(y_actual, y_predictions)
                    auc = roc_auc_score(y_actual, y_probabilities)
                    prec = precision_score(y_actual, y_predictions, average='binary', zero_division=0)
                    rec = recall_score(y_actual, y_predictions, average='binary', zero_division=0)
                    f1 = f1_score(y_actual, y_predictions, average='binary', zero_division=0)
                    mcc = matthews_corrcoef(y_actual, y_predictions)
                    
                    st.markdown("### 📊 Performance Metrics")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("🎯 Accuracy", f"{acc:.4f}", help="Overall prediction accuracy")
                        st.metric("📉 Recall", f"{rec:.4f}", help="Sensitivity / True Positive Rate")
                    
                    with metric_col2:
                        st.metric("📈 AUC Score", f"{auc:.4f}", help="Area Under ROC Curve")
                        st.metric("🎲 F1 Score", f"{f1:.4f}", help="Harmonic mean of Precision & Recall")
                    
                    with metric_col3:
                        st.metric("🔍 Precision", f"{prec:.4f}", help="Positive Predictive Value")
                        st.metric("🔗 MCC", f"{mcc:.4f}", help="Matthews Correlation Coefficient")
                    
                    st.markdown("---")
                    
                    # Visualization Section
                    st.markdown("### 📉 Detailed Analysis")
                    
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Prediction Distribution"])
                    
                    with viz_tab1:
                        # Confusion Matrix
                        cm = confusion_matrix(y_actual, y_predictions)
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Predicted: Average', 'Predicted: Good'],
                            y=['Actual: Average', 'Actual: Good'],
                            text=cm,
                            texttemplate='%{text}',
                            textfont={"size": 18, "color": "white"},
                            colorscale='Reds',
                            showscale=True,
                            hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title=f'Confusion Matrix - {selected_classifier}',
                            xaxis_title='Predicted Class',
                            yaxis_title='Actual Class',
                            height=450,
                            font=dict(size=12)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        cm_detail_col1, cm_detail_col2 = st.columns(2)
                        with cm_detail_col1:
                            st.metric("True Negatives", int(cm[0, 0]), help="Correctly predicted Average wines")
                            st.metric("False Positives", int(cm[0, 1]), help="Average wines predicted as Good")
                        with cm_detail_col2:
                            st.metric("False Negatives", int(cm[1, 0]), help="Good wines predicted as Average")
                            st.metric("True Positives", int(cm[1, 1]), help="Correctly predicted Good wines")
                    
                    with viz_tab2:
                        # Classification Report
                        report = classification_report(y_actual, y_predictions, 
                                                      target_names=['Average Wine', 'Good Wine'],
                                                      output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        
                        st.dataframe(
                            report_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', subset=['f1-score']),
                            use_container_width=True
                        )
                    
                    with viz_tab3:
                        # Prediction Distribution
                        dist_df = pd.DataFrame({
                            'Actual': y_actual.map({0: 'Average Wine', 1: 'Good Wine'}),
                            'Predicted': pd.Series(y_predictions).map({0: 'Average Wine', 1: 'Good Wine'})
                        })
                        
                        fig = px.histogram(dist_df, x='Predicted', color='Actual',
                                          barmode='group',
                                          title=f'Prediction Distribution - {selected_classifier}',
                                          color_discrete_map={'Average Wine': '#e74c3c', 'Good Wine': '#27ae60'},
                                          labels={'count': 'Number of Samples'})
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # CASE 2: WITHOUT LABELS (PREDICTION ONLY MODE)
                else:
                    st.success("✅ Predictions generated successfully!")
                    
                    st.markdown("### 🔮 Prediction Results")
                    
                    results_df = pd.DataFrame({
                        'Sample ID': range(1, len(y_predictions) + 1),
                        'Predicted Class': pd.Series(y_predictions).map({0: 'Average Wine', 1: 'Good Wine'}),
                        'Confidence Score': y_probabilities
                    })
                    
                    st.dataframe(results_df, use_container_width=True, height=400)
                    
                    # Prediction summary
                    summary_col1, summary_col2 = st.columns(2)
                    summary_col1.metric("Predicted Good Wines", int(y_predictions.sum()))
                    summary_col2.metric("Predicted Average Wines", int((y_predictions == 0).sum()))
                    
                    # Distribution chart
                    fig = px.pie(
                        values=[int((y_predictions == 0).sum()), int(y_predictions.sum())],
                        names=['Average Wine', 'Good Wine'],
                        title='Prediction Distribution',
                        color_discrete_sequence=['#e74c3c', '#27ae60']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the correct format and features.")
    
    else:
        st.info("👆 Please upload a CSV file to begin wine quality prediction")
        
        st.markdown("### 📋 Expected CSV Format")
        st.markdown("""
        Your CSV should contain the following **12 features**:
        
        | Feature | Description | Unit |
        |---------|-------------|------|
        | fixed acidity | Fixed acidity level | g/dm³ |
        | volatile acidity | Volatile acidity level | g/dm³ |
        | citric acid | Citric acid content | g/dm³ |
        | residual sugar | Residual sugar amount | g/dm³ |
        | chlorides | Chloride content | g/dm³ |
        | free sulfur dioxide | Free SO₂ level | mg/dm³ |
        | total sulfur dioxide | Total SO₂ level | mg/dm³ |
        | density | Wine density | g/cm³ |
        | pH | pH value | - |
        | sulphates | Sulphate content | g/dm³ |
        | alcohol | Alcohol percentage | % vol |
        | wine_type | Wine type (0=White, 1=Red) | binary |
        | quality_class | Target (0=Average, 1=Good) | *optional* |
        """)

# TAB 2: MODEL PERFORMANCE
with tab2:
    st.subheader("📈 Comparative Model Performance Analysis")
    
    try:
        performance_df = pd.read_csv('model/wine_model_performance.csv')
        
        st.markdown("### 📊 All Models Comparison Table")
        st.dataframe(
            performance_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'AUC', 'F1']),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Performance Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Bar chart for all metrics
            fig = go.Figure()
            metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
            
            for idx, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=performance_df['Model'],
                    y=performance_df[metric],
                    marker_color=colors[idx]
                ))
            
            fig.update_layout(
                title='Metrics Comparison Across All Models',
                barmode='group',
                height=450,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            # Radar chart for best model
            best_model_idx = performance_df['Accuracy'].idxmax()
            best_model_data = performance_df.iloc[best_model_idx]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[best_model_data['Accuracy'], best_model_data['AUC'], 
                   best_model_data['Precision'], best_model_data['Recall'], 
                   best_model_data['F1'], best_model_data['MCC']],
                theta=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                fill='toself',
                name=best_model_data['Model'],
                line_color='#722F37'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f'Best Model: {best_model_data["Model"]}',
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model rankings
        st.markdown("### 🏆 Model Rankings")
        rank_col1, rank_col2, rank_col3 = st.columns(3)
        
        with rank_col1:
            st.markdown("**🥇 Best Accuracy**")
            best_acc = performance_df.nlargest(3, 'Accuracy')
            for idx, row in best_acc.iterrows():
                st.write(f"{row['Model']}: {row['Accuracy']:.4f}")
        
        with rank_col2:
            st.markdown("**🥇 Best AUC**")
            best_auc = performance_df.nlargest(3, 'AUC')
            for idx, row in best_auc.iterrows():
                st.write(f"{row['Model']}: {row['AUC']:.4f}")
        
        with rank_col3:
            st.markdown("**🥇 Best F1**")
            best_f1 = performance_df.nlargest(3, 'F1')
            for idx, row in best_f1.iterrows():
                st.write(f"{row['Model']}: {row['F1']:.4f}")
    
    except FileNotFoundError:
        st.warning("⚠️ Performance data not found. Please train models first.")

# TAB 3: ABOUT DATASET
with tab3:
    st.subheader("ℹ️ Wine Quality Dataset Information")
    
    st.markdown("""
    ### 📚 Dataset Overview
    
    This application uses the **Wine Quality Dataset** from the UCI Machine Learning Repository,
    which contains physicochemical and sensory data for red and white Portuguese "Vinho Verde" wines.
    
    ### 🎯 Classification Task
    
    - **Type**: Binary Classification
    - **Target Variable**: Wine quality class
      - **Class 0**: Average Wine (original quality score < 6)
      - **Class 1**: Good Wine (original quality score ≥ 6)
    
    ### 📊 Dataset Statistics
    
    - **Total Samples**: 6,497 wines (1,599 red + 4,898 white)
    - **Features**: 12 physicochemical properties
    - **Source**: UCI ML Repository
    - **Domain**: Wine Chemistry & Quality Assessment
    
    ### 🔬 Feature Descriptions
    """)
    
    feature_details = pd.DataFrame({
        'Feature': [
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol', 'wine_type'
        ],
        'Description': [
            'Tartaric acid content (non-volatile acids)',
            'Acetic acid content (can lead to vinegar taste)',
            'Adds freshness and flavor to wines',
            'Sugar remaining after fermentation',
            'Salt content in wine',
            'Prevents microbial growth and oxidation',
            'Total amount of SO2 (free + bound)',
            'Depends on alcohol and sugar content',
            'Acidity/basicity level (0-14 scale)',
            'Wine additive contributing to SO2 levels',
            'Alcohol percentage by volume',
            'Binary indicator (0=White, 1=Red)'
        ],
        'Unit': [
            'g/dm³', 'g/dm³', 'g/dm³', 'g/dm³', 'g/dm³',
            'mg/dm³', 'mg/dm³', 'g/cm³', 'scale', 'g/dm³',
            '% vol', 'binary'
        ]
    })
    
    st.dataframe(feature_details, use_container_width=True, height=460)
    
    st.markdown("""
    ### 🔗 References
    
    - **Original Dataset**: [UCI ML Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
    - **Citation**: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
      *Modeling wine preferences by data mining from physicochemical properties.*
      Decision Support Systems, Elsevier, 47(4):547-553, 2009.
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
    <p style='color: #495057; margin: 5px;'><strong>Wine Quality Classification System</strong></p>
    <p style='color: #6c757d; margin: 5px; font-size: 0.9rem;'>Machine Learning Assignment 2 | M.Tech (AIML/DSE)</p>
    <p style='color: #6c757d; margin: 5px; font-size: 0.9rem;'>BITS Pilani </p>
    <p style='color: #adb5bd; margin: 10px 0 5px 0; font-size: 0.85rem;'>Developed with Streamlit</p>
</div>
""", unsafe_allow_html=True)
