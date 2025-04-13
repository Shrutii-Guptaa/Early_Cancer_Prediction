# Early-Stage Cancer Prediction (Cervical Cancer Dataset)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

# Load and clean dataset
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    df = df.replace('?', np.nan)
    df = df.dropna()
    df = df.astype(float)
    return df

# Encode categorical variables 
def encode_categorical(df):
    label_encoders = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# Visualizations
def plot_correlation(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

def plot_distribution(df, column):
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

def plot_risk_by_feature(df, feature):
    if feature in df.columns and 'Biopsy' in df.columns:
        fig = px.histogram(df, x=feature, color='Biopsy', barmode='group', title=f"{feature} vs Biopsy Result")
        fig.show()

# Train models
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "KMeans": KMeans(n_clusters=2),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# Evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    if y_prob is not None:
        print("Log Loss:", log_loss(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Main pipeline
def main():
    df = load_and_clean_data("risk_factors_cervical_cancer.csv")
    df, encoders = encode_categorical(df)

    # Use 'Biopsy' as target column
    target_col = 'Biopsy'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    plot_correlation(df)
    plot_distribution(df, df.columns[0])
    plot_risk_by_feature(df, 'Age')

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = train_models(X_train, y_train)

    for name, model in models.items():
        print(f"--- {name} ---")
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
