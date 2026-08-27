"""
train.py
Trains multiple classification models on the processed heart disease
dataset, compares their performance, and saves the best model + scaler
to the models/ directory. Also saves evaluation plots to results/.

Usage
-----
    python src/train.py
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

from data_preprocessing import run_pipeline, PROCESSED_PATH
from utils import TARGET_COL, evaluate_predictions

RANDOM_STATE = 42
MODELS_DIR = "models"
RESULTS_DIR = "results"


def main():
    # 1. Load / build processed data
    if os.path.exists(PROCESSED_PATH):
        df = pd.read_csv(PROCESSED_PATH)
    else:
        df = run_pipeline()

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # 2. Train/test split + scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # 3. Cross-validation (sanity check across models)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
    }
    print("\n--- 5-Fold Cross-Validation ---")
    for name, model in cv_models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_strategy, scoring="accuracy")
        print(f"{name:20s}  Mean CV Accuracy: {scores.mean():.4f}  (+/- {scores.std():.4f})")

    # 4. Train final models
    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
    knn = KNeighborsClassifier(n_neighbors=9).fit(X_train_scaled, y_train)
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
    dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                         eval_metric="logloss", random_state=RANDOM_STATE).fit(X_train_scaled, y_train)

    voting_clf = VotingClassifier(
        estimators=[("lr", log_reg), ("rf", rf), ("xgb", xgb)], voting="soft"
    ).fit(X_train_scaled, y_train)

    trained_models = {
        "Logistic Regression": log_reg,
        "KNN": knn,
        "SVM": svm,
        "Decision Tree": dt,
        "Random Forest": rf,
        "XGBoost": xgb,
        "Voting Ensemble": voting_clf,
    }

    # 5. Evaluate on the held-out test set
    print("\n--- Test Set Evaluation ---")
    results_table = []
    predictions = {}
    for name, model in trained_models.items():
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred
        metrics = evaluate_predictions(y_test, y_pred)
        metrics["Model"] = name
        results_table.append(metrics)
        print(f"{name:20s}  Accuracy: {metrics['Accuracy']:.4f}  F1: {metrics['F1-Score']:.4f}")

    results_df = pd.DataFrame(results_table).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest performing model: {best_model_name} (Accuracy: {results_df.iloc[0]['Accuracy']:.4f})")

    # 6. Save metrics table
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(results_df.to_string(index=False))

    # 7. Confusion matrix plot
    cm = confusion_matrix(y_test, predictions[best_model_name])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix — {best_model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # 8. ROC curve plot
    plt.figure(figsize=(8, 6))
    roc_models = {"Logistic Regression": log_reg, "Random Forest": rf, "XGBoost": xgb, "Voting Ensemble": voting_clf}
    for name, model in roc_models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    print(f"\nEvaluation plots saved to: {RESULTS_DIR}/")

    # 9. Save best model + scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))
    print(f"Best model ('{best_model_name}') and scaler saved to: {MODELS_DIR}/")

    # 10. Print full classification report for the best model
    print("\n--- Classification Report (Best Model) ---")
    print(classification_report(y_test, predictions[best_model_name], target_names=["No Disease", "Disease"]))


if __name__ == "__main__":
    main()
