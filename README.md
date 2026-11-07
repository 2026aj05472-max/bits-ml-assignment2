# End-to-End Machine Learning Classification & Streamlit Deployment
**BITS Pilani - Work Integrated Learning Programmes Division**  
**Course:** M.Tech (AIML) - Machine Learning | **Assignment - 2**
**BITSID:2026aj05472   NAME:MODIBOYANA MAHIDHAR**

---

## 1. Problem Statement
The objective of this assignment is to develop, evaluate, and deploy five distinct supervised machine learning classification algorithms to predict breast tumor diagnosis (Malignant vs. Benign). In medical diagnostics, minimizing false negatives (high recall for malignant cases) while preserving high diagnostic accuracy and precision is critical. An interactive Streamlit web dashboard is deployed to enable real-time model selection, test data upload, evaluation metric inspection, and confusion matrix visualization.

---

## 2. Dataset Description
- **Dataset Chosen:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI Machine Learning Repository / Scikit-Learn)
- **Total Number of Instances:** 569 instances (Exceeds minimum requirement of 500)
- **Total Number of Features:** 30 continuous real-valued features (Exceeds minimum requirement of 12)
- **Target Variable:** Binary classification (`0` = Malignant, `1` = Benign)
- **Feature Attributes:** Derived from digitized images of fine needle aspirate (FNA) of breast masses, describing characteristics of the cell nuclei:
  - Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave points, Symmetry, Fractal dimension (calculated for Mean, Standard Error, and Worst/Largest value).
- **Split Ratio:** 80% Training (455 samples) and 20% Holdout Testing (114 samples) using stratified sampling.

---

## 3. Mandatory Links
- **GitHub Repository:** https://github.com/2026aj05472-max/bits-ml-assignment2
- **Live Streamlit App:** https://its-ml-assignment2-2026aj05472.streamlit.app

---

## 4. Models Used & Evaluation Metrics Comparison Table
All five models were trained using the same 80% training split and evaluated strictly on the untouched holdout `test_data.csv` (114 instances).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

---

## 5. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved outstanding performance with an Accuracy of 0.9825, AUC of 0.9954, and MCC of 0.9623. Standardizing features allowed the linear decision boundary with L2 regularization to generalize exceptionally well without overfitting the 30 continuous clinical features. |
| **Decision Tree** | Yielded an Accuracy of 0.9211 and AUC of 0.9163. While highly interpretable via orthogonal decision rules, single decision trees tend to suffer from high variance and axis-aligned boundaries, leading to slightly lower sensitivity on subtle boundary patterns. |
| **KNN** | Produced strong predictive accuracy (0.9561) and F1-score (0.9655) with $k=5$. Because distance metrics (Euclidean) are sensitive to dimensionality, feature scaling was essential to prevent higher-magnitude attributes from dominating neighbor voting. |
| **Naive Bayes** | Reached an Accuracy of 0.9386 and a remarkably high AUC of 0.9878 despite assuming conditional independence between features. While clinical cell measurements exhibit high multi-collinearity, Gaussian Naive Bayes remains a robust, low-variance baseline. |
| **Support Vector Machine** | Tied for the top performance with an Accuracy of 0.9825 and an MCC of 0.9623, boasting an AUC of 0.9950. The RBF kernel effectively projected the high-dimensional feature space into an optimal separating hyperplane with maximum margin separation. |
| **Overall Winner for your dataset?** | **Logistic Regression & Support Vector Machine (Tie)**. Both models demonstrated superior discriminative power, achieving the highest MCC (0.9623), F1 (0.9861), and Accuracy (0.9825). Logistic Regression holds a slight practical edge due to computational efficiency and native probability calibration. |

---

## 6. Repository Structure
