from src.preprocessing import preprocess_data
from src.models.logistic_regression import train_model as train_logistic
from src.models.decision_tree import train_model as train_tree

from src.models.visualization import generate_all_visualizations


# =========================================================
# Load and split data
# =========================================================

X_train, X_test, y_train, y_test = preprocess_data()


# =========================================================
# Logistic Regression
# =========================================================

print("\nTraining Logistic Regression...")

logistic_model = train_logistic(
    X_train,
    y_train
)


# =========================================================
# Decision Tree
# =========================================================

print("\nTraining Decision Tree...")

tree_model = train_tree(
    X_train,
    y_train
)


# =========================================================
# Generate all visualizations
# =========================================================

print("\nGenerating Model Visualizations...")

generate_all_visualizations(
    logistic_model,
    tree_model,
    X_test,
    y_test,
    X_train.columns
)