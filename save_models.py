"""
save_models.py
Run this once to train and save svm.pkl, decision_tree.pkl, and logistic_regression.pkl
into the models/ directory (random_forest.pkl is already there).

Usage:
    uv run python save_models.py
    -- or --
    python save_models.py
"""

import sys
import os

# Make sure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
from preprocessing import preprocess_data
from models.svm import svm_model
from models.decision_tree import train_model as train_dt
from models.logistic_regression import train_model as train_lr

# ── 1. Load & split data ────────────────────────────────────────────────────
print("Loading and splitting data...")
X_train, X_test, y_train, y_test = preprocess_data()

# ── 2. SVM ───────────────────────────────────────────────────────────────────
print("\nTraining SVM (GridSearchCV – may take a minute)...")
svm = svm_model(X_train, y_train)
joblib.dump(svm, "models/svm.pkl")
print("✅  SVM model saved → models/svm.pkl")

# ── 3. Decision Tree ─────────────────────────────────────────────────────────
print("\nTraining Decision Tree (GridSearchCV)...")
dt = train_dt(X_train, y_train)
joblib.dump(dt, "models/decision_tree.pkl")
print("✅  Decision Tree model saved → models/decision_tree.pkl")

# ── 4. Logistic Regression ───────────────────────────────────────────────────
print("\nTraining Logistic Regression...")
lr = train_lr(X_train, y_train)
joblib.dump(lr, "models/logistic_regression.pkl")
print("✅  Logistic Regression model saved → models/logistic_regression.pkl")

print("\nAll models saved successfully!")
