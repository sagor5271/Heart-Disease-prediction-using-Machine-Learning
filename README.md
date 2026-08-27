# ❤️ Heart Disease Prediction using Machine Learning

A machine learning project that predicts the likelihood of heart disease in a patient based on clinical and diagnostic features, using classic supervised learning algorithms and an ensemble model.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

---

## 📌 Overview

Cardiovascular disease is one of the leading causes of death worldwide, and early prediction can significantly improve patient outcomes. This project builds and compares multiple ML classifiers to predict whether a patient is likely to have heart disease based on clinical parameters such as age, blood pressure, cholesterol, and exercise-related measurements.

## 📊 Dataset

- **File:** `data/raw/heart.csv`
- **Records:** 918 patients
- **Features:** 11 clinical attributes + 1 binary target (`HeartDisease`)

| Feature | Description |
|---|---|
| Age | Age of the patient |
| Sex | M / F |
| ChestPainType | TA, ATA, NAP, ASY |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl (1/0) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y/N) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Up, Flat, Down |
| **HeartDisease** | **Target** — 1 = disease present, 0 = normal |

## 🧠 Methodology

1. **Preprocessing** — duplicate removal, invalid-zero handling (Cholesterol/RestingBP), median imputation, label encoding
2. **EDA** — target balance, distributions, correlation heatmap (see `notebooks/EDA.ipynb`)
3. **Outlier removal** — IQR method on RestingBP and Cholesterol
4. **Modeling** — Logistic Regression, KNN, SVM, Decision Tree, Random Forest, XGBoost, and a soft-voting Ensemble
5. **Evaluation** — 5-fold stratified cross-validation, accuracy/precision/recall/F1, confusion matrix, ROC-AUC
6. **Feature selection** — Random Forest feature importance

## 🔍 Exploratory Data Analysis

**Age distribution by heart disease status** — disease prevalence rises noticeably after age 50.

<img width="609" height="470" alt="image" src="https://github.com/user-attachments/assets/2ebf900c-16ca-4b1a-b01c-5e943f00c79d" />


**Categorical features vs. target** — sex, chest pain type, exercise-induced angina, and ST slope all show strong separation between classes.

<img width="1389" height="989" alt="image" src="https://github.com/user-attachments/assets/2a0cf3fd-ddd5-4222-9537-79b26b827879" />


**Numeric feature distributions by target class**

<img width="2190" height="490" alt="image" src="https://github.com/user-attachments/assets/ab0700fc-5b7a-42a0-a34d-596ce1f79c64" />


**Feature correlation heatmap** — `ST_Slope`, `ExerciseAngina`, and `Oldpeak` show the strongest correlation with `HeartDisease`.

<img width="868" height="774" alt="image" src="https://github.com/user-attachments/assets/6368ecc9-b445-49e6-a1d5-5cb571198d9b" />


## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **KNN** | **0.895** | 0.896 | 0.915 | 0.905 |
| SVM | 0.895 | 0.904 | 0.904 | 0.904 |
| Random Forest | 0.889 | 0.903 | 0.894 | 0.898 |
| Voting Ensemble | 0.889 | 0.903 | 0.894 | 0.898 |
| Logistic Regression | 0.883 | 0.911 | 0.872 | 0.891 |
| XGBoost | 0.883 | 0.902 | 0.883 | 0.892 |
| Decision Tree | 0.848 | 0.878 | 0.840 | 0.859 |

*(Full run available in `results/metrics.csv` and `notebooks/model_training.ipynb`)*

**Model performance comparison**

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/b2712c7e-403d-455e-9937-6c18afdea96e" />


**5-fold cross-validation accuracy by model**

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/4b099c3b-7d10-456d-8563-e7985c20cf16" />


**Confusion matrix — best model (KNN)**

<img width="435" height="393" alt="image" src="https://github.com/user-attachments/assets/7a5ed8a6-08df-40d6-a8d1-87f5d84f4a59" />


**ROC curve comparison**

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/e134fa01-3b04-4a45-9c08-5e5f3f038a92" />


**Feature importance (Random Forest)** — `ST_Slope`, `ChestPainType`, and `Oldpeak` are the most influential predictors.

<img width="889" height="590" alt="feature_importance_rf" src="https://github.com/user-attachments/assets/d26d2436-f41a-4a79-a51e-33c5d2ed1f44" />


## 🛠️ Tech Stack

Python 3.9+ · pandas · numpy · scikit-learn · xgboost · matplotlib · seaborn · Streamlit

## 📂 Project Structure

```
Heart-Disease-Prediction-ML/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw/heart.csv
│   └── processed/heart_processed.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── results/
│   ├── age_distribution.png
│   ├── categorical_vs_target.png
│   ├── feature_boxplots.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   ├── cv_accuracy_boxplot.png
│   ├── confusion_matrix_knn.png
│   ├── roc_curve.png
│   ├── feature_importance_rf.png
│   └── metrics.csv / metrics.txt
│
└── app/
    └── streamlit_app.py
```

## ⚙️ Installation & Usage

```bash
git clone https://github.com/sagor5271/Heart-Disease-prediction-using-Machine-Learning.git
cd Heart-Disease-prediction-using-Machine-Learning
pip install -r requirements.txt
```

**Run the full pipeline (preprocessing → training → evaluation):**
```bash
python src/train.py
```
This regenerates `data/processed/heart_processed.csv`, `models/*.pkl`, and `results/*`.

**Predict on new patient data:**
```bash
python src/predict.py --input sample_input.csv --output predictions.csv
```

**Launch the interactive demo app:**
```bash
streamlit run app/streamlit_app.py
```

**Or explore interactively:**
Open `notebooks/EDA.ipynb` for data exploration, or `notebooks/model_training.ipynb` for the full end-to-end pipeline.

## 🚀 Future Work

- Hyperparameter tuning (GridSearchCV / Optuna)
- Deploying the Streamlit app publicly (e.g., Streamlit Community Cloud)
- Testing on larger, more diverse clinical datasets

## 👤 Author

**Md Sagor Hossain**
Biomedical Engineering, Islamic University, Bangladesh
📧 sagor.bme.iu@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/md-sagor-hossain-8471ab336)

## 📄 License

MIT License — see [LICENSE](LICENSE).
