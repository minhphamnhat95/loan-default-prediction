**Loan Default Prediction**

Predicting customer loan default risk using machine learning and financial feature engineering.

**Project Overview**
Built an end-to-end classification pipeline on 35,000+ customer financial profiles to identify borrowers at risk of default. The project focuses on data preprocessing, feature engineering, model optimization, and comparative evaluation of multiple machine learning algorithms.

**Tech Stack**
- Python (Pandas, NumPy, Scikit-learn, Matplotlib)

**Key Features**
- Prevented data leakage through train/test-aware preprocessing
- Engineered financial risk indicators including:
  - Expense Ratio
  - Savings Ratio
  - Loan Ratio
  - Financial Stress Score
  - High-Risk Borrower Flag

**Data Preprocessing:**
  - Missing value imputation
  - Log transformation
  - Binning
  - One-hot encoding
  - Feature scaling

**Models Evaluated**
- Model	F1 Score	AUC
- Decision Tree	0.794	0.852
- Random Forest	0.771	0.854
- Neural Network	0.753	0.846
- SVM	0.730	0.836
- KNN	0.709	0.828

**Results**
The Decision Tree achieved the best overall performance:
  - Accuracy: 84.5%
  - Precision: 81.5%
  - Recall: 77.4%
  - F1 Score: 79.4%
The project demonstrated that domain-specific financial risk features significantly improved borrower risk separation and model performance.

**Key Learnings**
- Machine learning model development
- Financial risk analytics
- Feature engineering for tabular data
- Hyperparameter tuning with GridSearchCV
- Cross-validation and model evaluation
- Model interpretability vs complexity trade-offs
