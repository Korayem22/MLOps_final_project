import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# ============ Setup ============ #
mlflow.set_experiment("Hand Gesture Classification")
os.makedirs("artifacts", exist_ok=True)

# ============ Load and Prepare Data ============ #
df = pd.read_csv('hand_landmarks_data.csv')

def normalize_landmarks(row):
    wrist_x, wrist_y = row["x1"], row["y1"]
    mid_x, mid_y = row["x13"], row["y13"]
    scale = np.sqrt((mid_x - wrist_x)**2 + (mid_y - wrist_y)**2)
    scale = 1 if scale == 0 else scale
    for i in range(21):
        row[f"x{i+1}"] = (row[f"x{i+1}"] - wrist_x) / scale
        row[f"y{i+1}"] = (row[f"y{i+1}"] - wrist_y) / scale
    return row

df_normalized = df.apply(normalize_landmarks, axis=1)

# Label encoding
label_encoder = LabelEncoder()
df_normalized["label_encoded"] = label_encoder.fit_transform(df_normalized["label"])
df_normalized.drop(columns=["label"], inplace=True)

# Save label encoder
joblib.dump(label_encoder, "artifacts/label_encoder.pkl")

# Features and labels
features = df_normalized.drop(["label_encoded"], axis=1)
labels = df_normalized["label_encoded"]

# Splitting
features_train, features_validation_test, labels_train, labels_validation_test = train_test_split(
    features, labels, test_size=0.4, random_state=100)
features_validation, features_test, labels_validation, labels_test = train_test_split(
    features_validation_test, labels_validation_test, test_size=0.5, random_state=100)

# ============ Model Training and Logging ============ #
models = {
    "LogisticRegression": LogisticRegression(multi_class="multinomial"),
    "DecisionTree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

best_f1 = 0
best_model = None
best_model_name = ""

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(features_train, labels_train)
        predictions = model.predict(features_validation)
        proba = model.predict_proba(features_validation) if hasattr(model, "predict_proba") else None

        # Metrics
        acc = accuracy_score(labels_validation, predictions)
        prec = precision_score(labels_validation, predictions, average='weighted')
        rec = recall_score(labels_validation, predictions, average='weighted')
        f1 = f1_score(labels_validation, predictions, average='weighted')

        # Logging
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Signature inference for model inputs and outputs
        sample_input = features_validation.head(5)
        sample_output = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        # Log model with signature
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=sample_input
        )

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name
            joblib.dump(model, "artifacts/rf_model.pkl")
            mlflow.log_artifact("artifacts/rf_model.pkl")
            mlflow.log_artifact("artifacts/label_encoder.pkl")

print(f"Best model: {best_model_name} with F1 Score: {best_f1:.3f}")
