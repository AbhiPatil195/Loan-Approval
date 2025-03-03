#!/usr/bin/env python
# coding: utf-8

# Loan Approval Prediction Model
# This script predicts loan approval status based on various applicant features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Comment out imbalanced-learn temporarily
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import warnings
import os
import joblib
from pathlib import Path

# Set up paths
DATA_DIR = Path(r"C:\Users\lenovo\Downloads\playground-series-s4e10")
TRAIN_PATH = DATA_DIR / "playground-series-s4e10 (2)" / "train.csv"
TEST_PATH = DATA_DIR / "playground-series-s4e10" / "test.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR = Path("output")
MODEL_PATH = OUTPUT_DIR / "loan_model.pkl"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

def load_data():
    """Load the datasets"""
    print("Loading datasets...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df_sub = pd.read_csv(SUBMISSION_PATH)
    
    print(f"Training data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
    
    return df_train, df_test, df_sub

def explore_data(df_train):
    """Perform basic EDA and visualizations"""
    print("\nExploring training data...")
    print("\nBasic information:")
    print(df_train.info())
    
    print("\nSummary statistics:")
    print(df_train.describe())
    
    print("\nChecking for missing values:")
    print(df_train.isnull().sum())
    
    # Plot target variable distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df_train["loan_status"])
    plt.title("Loan Status Distribution")
    plt.savefig(OUTPUT_DIR / "loan_status_distribution.png")
    
    # We'll skip the correlation matrix for now since it requires numeric data
    print("\nPlots saved to output directory.")
    
    return df_train

def encode_and_show_correlation(df):
    """Encode categorical features and show correlation matrix"""
    print("\nEncoding categorical features for correlation analysis...")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_encoded = df.copy()
    
    # Encode categorical features
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    label_enc = LabelEncoder()
    
    for col in categorical_cols:
        df_encoded[col] = label_enc.fit_transform(df_encoded[col])
    
    # Generate correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df_encoded.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.2)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
    print("Correlation matrix saved to output directory.")

def preprocess_data(df_train, df_test):
    """Preprocess and transform data for modeling"""
    print("\nPreprocessing data...")
    
    # Handle missing values
    df_train['person_emp_length'] = df_train['person_emp_length'].fillna(df_train['person_emp_length'].mean())
    df_train['loan_int_rate'] = df_train['loan_int_rate'].fillna(df_train['loan_int_rate'].mean())
    
    # Generate and save correlation matrix after filling missing values
    encode_and_show_correlation(df_train)
    
    # Drop ID column
    df_train = df_train.drop(columns=['id'])
    df_test_ids = df_test['id'].copy()  # Save IDs for later
    df_test = df_test.drop(columns=['id'])
    
    # Encode categorical features
    label_enc = LabelEncoder()   
    label_cols = ['person_home_ownership', 'loan_grade', 'cb_person_default_on_file']    

    for col in label_cols:
        df_train[col] = label_enc.fit_transform(df_train[col])
        df_test[col] = label_enc.transform(df_test[col])    

    # One-hot encode loan_intent
    df_train = pd.get_dummies(df_train, columns=['loan_intent'], drop_first=True)
    df_test = pd.get_dummies(df_test, columns=['loan_intent'], drop_first=True) 
    
    # Ensure test data has same columns as train data
    train_columns = df_train.columns.drop('loan_status') if 'loan_status' in df_train.columns else df_train.columns
    missing_cols = set(train_columns) - set(df_test.columns)
    
    for col in missing_cols:
        df_test[col] = 0
        
    df_test = df_test[train_columns]
    
    return df_train, df_test, df_test_ids

def feature_engineering(df_train, df_test):
    """Create new features to improve model performance"""
    print("\nPerforming feature engineering...")
    
    for df in [df_train, df_test]:
        # Create new features
        df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']  
        df['financial_burden'] = df['loan_amnt'] * df['loan_int_rate'] 
        df['income_per_year_emp'] = df['person_income'] / (df['person_emp_length'] + 1e-5)
        df['cred_hist_to_age_ratio'] = df['cb_person_cred_hist_length'] / df['person_age']
        df['int_to_loan_ratio'] = df['loan_int_rate'] / df['loan_amnt']
        df['loan_int_emp_interaction'] = df['loan_int_rate'] * df['person_emp_length']
        df['debt_to_credit_ratio'] = df['loan_amnt'] / (df['cb_person_cred_hist_length'] + 1e-5)
        df['int_to_cred_hist'] = df['loan_int_rate'] / (df['cb_person_cred_hist_length'] + 1e-5)
        df['int_per_year_emp'] = df['loan_int_rate'] / (df['person_emp_length'] + 1e-5)
        df['loan_amt_per_emp_year'] = df['loan_amnt'] / (df['person_emp_length'] + 1e-5)
        df['income_to_loan_ratio'] = df['person_income'] / (df['loan_amnt'] + 1e-5)
        
        # Handle inf/na values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mean(), inplace=True)
    
    print(f"Training data shape after feature engineering: {df_train.shape}")
    print(f"Test data shape after feature engineering: {df_test.shape}")
    return df_train, df_test

def train_model(df_train):
    """Train and evaluate the model"""
    print("\nPreparing data for training...")
    
    # Split data into features and target
    X = df_train.drop(columns=["loan_status"])
    y = df_train["loan_status"]
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert back to DataFrame to maintain column names
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    
    # Handle class imbalance using class weights instead of SMOTE
    print("\nHandling class imbalance using class weights...")
    
    # Create and train logistic regression model
    print("\nTraining logistic regression model...")
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    
    # Grid search for hyperparameter optimization
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],  # Updated to use only l2 which is compatible with all solvers
        'solver': ['lbfgs', 'liblinear']  # Updated solvers
    }
    
    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"\nBest parameters: {grid.best_params_}")
    
    # Evaluate the model
    y_probs = best_model.predict_proba(X_val)[:, 1]
    y_preds = best_model.predict(X_val)
    
    print("\nModel Evaluation on Validation Set:")
    print(classification_report(y_val, y_preds))
    print(f"ROC AUC Score: {roc_auc_score(y_val, y_probs):.4f}")
    
    # Save the model and scaler
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    
    return best_model, scaler

def make_predictions(df_test, df_test_ids, model, scaler):
    """Generate predictions on test data"""
    print("\nGenerating predictions on test data...")
    
    # Standardize test data
    X_test_scaled = scaler.transform(df_test)
    
    # Predict
    predictions = model.predict(X_test_scaled)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': df_test_ids,
        'loan_status': predictions
    })
    
    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Predictions saved to {submission_path}")
    
    return submission

def main():
    """Main function to orchestrate the loan prediction workflow"""
    print("Starting Loan Approval Prediction Model")
    
    # Load data
    df_train, df_test, df_sub = load_data()
    
    # Explore data
    df_train = explore_data(df_train)
    
    # Preprocess data
    df_train, df_test, df_test_ids = preprocess_data(df_train, df_test)
    
    # Feature engineering
    df_train, df_test = feature_engineering(df_train, df_test)
    
    # Train model
    model, scaler = train_model(df_train)
    
    # Make predictions
    submission = make_predictions(df_test, df_test_ids, model, scaler)
    
    print("\nLoan prediction process completed successfully!")

if __name__ == "__main__":
    main() 