import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             )
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('../bank/bank-full.csv', sep=';')

print(f"Dataset shape: {df.shape}")
print(f"\nDataset columns:\n{df.columns.tolist()}")
print(f"\nTarget distribution:\n{df['y'].value_counts()}")


X = df.drop('y', axis=1)
y = df['y']
y = y.map({'yes': 1, 'no': 0})

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_encoded = X.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing objects
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save test data for Streamlit app
test_df = X_test.copy()
test_df['y'] = y_test
test_df.to_csv('test_data.csv', index=False)
print("\nTest data saved as 'test_data.csv'")

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = y_pred
    
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    return metrics

results = []
models_dict = {}

print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

print("\n1. Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_metrics = evaluate_model(lr_model, X_test_scaled, y_test, 'Logistic Regression')
results.append(lr_metrics)
models_dict['Logistic Regression'] = lr_model
print(f"   Accuracy: {lr_metrics['Accuracy']:.4f}")

print("\n2. Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
dt_metrics = evaluate_model(dt_model, X_test, y_test, 'Decision Tree')
results.append(dt_metrics)
models_dict['Decision Tree'] = dt_model
print(f"   Accuracy: {dt_metrics['Accuracy']:.4f}")

print("\n3. Training K-Nearest Neighbors...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_metrics = evaluate_model(knn_model, X_test_scaled, y_test, 'K-Nearest Neighbors')
results.append(knn_metrics)
models_dict['K-Nearest Neighbors'] = knn_model
print(f"   Accuracy: {knn_metrics['Accuracy']:.4f}")

print("\n4. Training Naive Bayes (Gaussian)...")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_metrics = evaluate_model(nb_model, X_test_scaled, y_test, 'Naive Bayes')
results.append(nb_metrics)
models_dict['Naive Bayes'] = nb_model
print(f"   Accuracy: {nb_metrics['Accuracy']:.4f}")

print("\n5. Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
rf_metrics = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
results.append(rf_metrics)
models_dict['Random Forest'] = rf_model
print(f"   Accuracy: {rf_metrics['Accuracy']:.4f}")

print("\n6. Training XGBoost...")
xgb_model = XGBClassifier(random_state=42, max_depth=6, n_estimators=100, 
                          eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train, y_train)
xgb_metrics = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
results.append(xgb_metrics)
models_dict['XGBoost'] = xgb_model
print(f"   Accuracy: {xgb_metrics['Accuracy']:.4f}")

results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(results_df.to_string(index=False))

results_df.to_csv('model_results.csv', index=False)
print("\nResults saved to 'model_results.csv'")

for model_name, model in models_dict.items():
    filename = f"model_{model_name.replace(' ', '_').lower()}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {filename}")

print("TRAINING COMPLETE!")