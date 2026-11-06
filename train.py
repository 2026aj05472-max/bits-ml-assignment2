import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib

def main():
    os.makedirs("model", exist_ok=True)
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv("test_data.csv", index=False)
    
    pipelines = {
        "logistic_regression": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
        "decision_tree": Pipeline([('clf', DecisionTreeClassifier(random_state=42, max_depth=5))]),
        "knn": Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
        "naive_bayes": Pipeline([('clf', GaussianNB())]),
        "svm": Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=42))])
    }
    
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, f"model/{name}.joblib")
        
    print("Training finished and artifacts saved.")

if __name__ == "__main__":
    main()
