# Generated from: Streamlit_app.py.ipynb
# Converted at: 2026-04-21T19:20:54.778Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/Loan_Default.csv')
display(df.head())

df.info()

# ### Removing Null and Duplicate Rows


print(f"Original DataFrame shape: {df.shape}")

# Drop rows with any null values (just in case, after imputation)
initial_rows = df.shape[0]
df.dropna(inplace=True)
print(f"DataFrame shape after dropping nulls: {df.shape} ({initial_rows - df.shape[0]} rows removed)")

# Drop duplicate rows
duplicate_rows_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"DataFrame shape after dropping duplicates: {df.shape} ({duplicate_rows_before - df.duplicated().sum()} duplicate rows removed)")

print("\nFirst 5 rows of the DataFrame after cleaning:")
display(df.head())

display(df.describe())

cols_to_drop = [
    "ID", "year", "rate_of_interest", "Interest_rate_spread",
    "Upfront_charges", "Secured_by", "submission_of_application",
    "construction_type", "total_units"
]

df = df.drop(columns=cols_to_drop)

df.info()

df.to_csv('cleaned_loan_default.csv', index=False)
print('DataFrame saved to cleaned_loan_default.csv')

cleaned_df = pd.read_csv('cleaned_loan_default.csv')

df_encoded = pd.get_dummies(df, drop_first=True)

def convert_age_to_numeric(age_str):
    if pd.isna(age_str):
        return np.nan
    if isinstance(age_str, (int, float)):
        return age_str
    if 'years' in age_str:
        return float(age_str.replace(' years', ''))
    if '-' in age_str:
        try:
            lower, upper = map(int, age_str.split('-'))
            return (lower + upper) / 2
        except ValueError:
            return np.nan
    if '<25' in age_str:
        return 20  # Midpoint of 0-25
    if '>74' in age_str:
        return 75 # Representative value for ages above 74
    return np.nan

df['age'] = df['age'].apply(convert_age_to_numeric)
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)

df.groupby("Status").mean(numeric_only=True)

# This cell contains outdated model training code.
# The correct and consolidated model training is in cell d7aa08eb.
# Please execute cell d7aa08eb for model training and then cell f8e4d8d8 for evaluation.


# ### Fixing the single-class error by reloading data and imputing missing values


# Reload the original dataset to reset the DataFrame to its initial state
df = pd.read_csv('/content/Loan_Default.csv')
print(f"DataFrame reloaded. New shape: {df.shape}")

# --- Handle missing values by imputation ---

# Impute numerical columns with their median
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed numerical column '{col}' with median: {median_val}")

# Impute categorical columns with their mode
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Imputed categorical column '{col}' with mode: {mode_val}")

print("\nMissing values after imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Verify 'Status' column still has both classes
print("\nStatus column value counts after imputation:")
print(df['Status'].value_counts())

# --- Drop specified columns ---
cols_to_drop = [
    "ID", "year", "rate_of_interest", "Interest_rate_spread",
    "Upfront_charges", "Secured_by", "submission_of_application",
    "construction_type", "total_units"
]
df = df.drop(columns=cols_to_drop)
print(f"Columns dropped. New shape: {df.shape}")

# --- Convert 'age' column to numeric and fill NaNs ---
def convert_age_to_numeric(age_str):
    if pd.isna(age_str):
        return np.nan
    if isinstance(age_str, (int, float)):
        return age_str
    if 'years' in age_str:
        return float(age_str.replace(' years', ''))
    if '-' in age_str:
        try:
            lower, upper = map(int, age_str.split('-'))
            return (lower + upper) / 2
        except ValueError:
            return np.nan
    if '<25' in age_str:
        return 20  # Midpoint of 0-25
    if '>74' in age_str:
        return 75 # Representative value for ages above 74
    return np.nan

df['age'] = df['age'].apply(convert_age_to_numeric)
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)
print("Age column converted to numeric and NaNs filled.")

# --- Perform one-hot encoding ---
df_encoded = pd.get_dummies(df, drop_first=True)
print(f"DataFrame encoded. New shape: {df_encoded.shape}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np # Import numpy for checking inf values

# Consolidate all data preprocessing steps to ensure df is in the correct state
df = pd.read_csv('/content/Loan_Default.csv')

# Impute numerical columns with their median
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Impute categorical columns with their mode
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

# Drop specified columns
cols_to_drop = [
    "ID", "year", "rate_of_interest", "Interest_rate_spread",
    "Upfront_charges", "Secured_by", "submission_of_application",
    "construction_type", "total_units"
]
df = df.drop(columns=cols_to_drop)

# Convert 'age' column to numeric and fill NaNs
def convert_age_to_numeric(age_str):
    if pd.isna(age_str):
        return np.nan
    if isinstance(age_str, (int, float)):
        return age_str
    if 'years' in age_str:
        return float(age_str.replace(' years', ''))
    if '-' in age_str:
        try:
            lower, upper = map(int, age_str.split('-'))
            return (lower + upper) / 2
        except ValueError:
            return np.nan
    if '<25' in age_str:
        return 20  # Midpoint of 0-25
    if '>74' in age_str:
        return 75 # Representative value for ages above 74
    return np.nan

df['age'] = df['age'].apply(convert_age_to_numeric)
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)

# Perform one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)
print(f"DataFrame encoded. New shape: {df_encoded.shape}")

# Check for non-numeric columns in df_encoded and convert them
non_numeric_cols_df_encoded = df_encoded.select_dtypes(exclude=np.number).columns
if len(non_numeric_cols_df_encoded) > 0:
    print(f"Non-numeric columns found in df_encoded: {list(non_numeric_cols_df_encoded)}")
    for col in non_numeric_cols_df_encoded:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    # Fill any NaNs introduced by pd.to_numeric (e.g., if a column was 'non-numeric strings')
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))
    print(f"NaNs after converting non-numeric and filling: {df_encoded.isnull().sum().sum()}")

X = df_encoded.drop("Status", axis=1)
y = df_encoded["Status"]

# Drop rows where 'y' (Status) is NaN
# This ensures that train_test_split does not encounter NaN in the stratification target
valid_indices = y.dropna().index
X = X.loc[valid_indices]
y = y.loc[valid_indices]

# Check for NaNs and inf values in X before scaling
print(f"NaNs in X before scaling: {X.isnull().sum().sum()}")
print(f"Infs in X before scaling: {np.isinf(X).sum().sum()}")

# Sanity check to ensure 'y' has at least two classes
if y.nunique() < 2:
    raise ValueError(
        "The 'Status' column in the processed data contains only one class. "
        "Please ensure the data loading and preprocessing steps (especially handling of missing values "
        "and one-hot encoding) are correctly executed and result in a 'Status' column with at least two classes."
        "You might need to re-run all data preprocessing cells before this one."
    )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify numerical columns for scaling
numerical_cols = X_train.select_dtypes(include=np.number).columns

# Initialize and fit StandardScaler on the training data
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("\nValue counts of y_train after stratification:")
print(y_train.value_counts())
print("\nValue counts of y_test after stratification:")
print(y_test.value_counts())

model = LogisticRegression(max_iter=2000) # Increased max_iter for convergence
model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# ### Model Evaluation


from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# **Reasoning**:
# To evaluate the model's performance using an ROC curve and AUC score, I need to predict probabilities on the test set, calculate FPR, TPR, and AUC, and then plot the ROC curve.
# 
# 


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Predict probabilities for the positive class (class 1)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate FPR, TPR, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"AUC Score: {auc_score:.2f}")

# **Reasoning**:
# I need to create a Pandas DataFrame `y_pred_df` from the `y_pred_proba` array and name the column 'predicted_proba' as instructed. This will make the predicted probabilities accessible for further analysis.
# 
# 


y_pred_df = pd.DataFrame(y_pred_proba, columns=['predicted_proba'])
print("DataFrame y_pred_df created successfully.")
display(y_pred_df.head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Instantiate a RandomForestClassifier model
# Setting random_state for reproducibility
rf_model = RandomForestClassifier(random_state=42)

# 2. Train the RandomForestClassifier model
print("Training RandomForestClassifier model...")
rf_model.fit(X_train, y_train)
print("RandomForestClassifier model trained successfully!")

# 3. Make predictions on the X_test dataset
y_pred_rf = rf_model.predict(X_test)

# 4. Print a classification report
print("\nRandomForestClassifier - Classification Report:")
print(classification_report(y_test, y_pred_rf))

# 5. Print a confusion matrix
print("\nRandomForestClassifier - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

from imblearn.over_sampling import SMOTE
from collections import Counter

# 1. Instantiate SMOTE
smote = SMOTE(random_state=42)

# 2. Apply SMOTE to the training data
print("Original training set shape:", Counter(y_train))
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 3. Print the value counts of y_resampled to verify class balance
print("Resampled training set shape:", Counter(y_resampled))

new_applicant_data = {
    'loan_limit': ['cf'],
    'Gender': ['Male'],
    'approv_in_adv': ['pre'],
    'loan_type': ['type1'],
    'loan_purpose': ['p1'],
    'Credit_Worthiness': ['l1'],
    'open_credit': ['nopc'],
    'business_or_commercial': ['nob/c'],
    'loan_amount': [300000],
    'term': [360.0],
    'Neg_ammortization': ['not_neg'],
    'interest_only': ['not_int'],
    'lump_sum_payment': ['not_lpsm'],
    'property_value': [450000.0],
    'occupancy_type': ['pr'],
    'income': [7500.0],
    'credit_type': ['CRIF'],
    'Credit_Score': [720],
    'co-applicant_credit_type': ['EXP'],
    'age': [38.0],
    'LTV': [66.67],
    'Region': ['North'],
    'Security_Type': ['direct'],
    'Status': [0],
    'dtir1': [30.0]
}

new_applicant_df = pd.DataFrame(new_applicant_data)
print("Hypothetical new applicant DataFrame created successfully.")
display(new_applicant_df)