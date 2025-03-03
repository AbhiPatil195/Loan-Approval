# Loan Approval Prediction

This project aims to predict loan defaults using machine learning techniques. It implements a logistic regression model to classify loan applications as default or non-default based on various applicant features.

## Project Structure

- `loan_approval_prediction.py`: Main script that contains the entire workflow
- `requirements.txt`: List of required Python packages
- `output/`: Directory where model outputs, visualizations, and predictions are saved

## Features

- Data preprocessing and feature engineering
- Handling of class imbalance using SMOTE and undersampling
- Logistic regression model with hyperparameter optimization
- Model evaluation with classification metrics
- Visualization of data distributions and correlations

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## Setup

1. Create a virtual environment (optional but recommended):
```
python -m venv venv
```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the main script:
```
python loan_approval_prediction.py
```

This will:
1. Load and preprocess the data
2. Perform feature engineering
3. Train a logistic regression model with optimized hyperparameters
4. Evaluate the model performance
5. Generate predictions on the test dataset
6. Save outputs to the 'output' directory

## Pushing to GitHub

1. Initialize a new git repository:
```
git init
```

2. Add all files to the repository:
```
git add .
```

3. Commit the changes:
```
git commit -m "Initial commit: Loan approval prediction project"
```

4. Create a new repository on GitHub (https://github.com/new)

5. Link your local repository to the GitHub repository:
```
git remote add origin https://github.com/your-username/loan-approval-prediction.git
```

6. Push your code to GitHub:
```
git push -u origin main
``` 