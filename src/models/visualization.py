import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.tree import plot_tree


def generate_all_visualizations(
    logistic_model,
    tree_model,
    X_test,
    y_test,
    feature_names
):

    # Create one figure with 5 separate sections
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 42),
        gridspec_kw={
            "height_ratios": [1, 1, 1, 1.5, 1]
        }
    )

    # =========================================================
    # 1. Logistic Regression - Confusion Matrix
    # =========================================================

    y_pred_logistic = logistic_model.predict(X_test)

    cm_logistic = confusion_matrix(
        y_test,
        y_pred_logistic
    )

    sns.heatmap(
        cm_logistic,
        annot=True,
        fmt="d",
        cmap="viridis",
        ax=axes[0]
    )

    axes[0].set_title(
        "Logistic Regression - Confusion Matrix",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")


    # =========================================================
    # 2. Logistic Regression - ROC Curve
    # =========================================================

    y_probability = logistic_model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_auc = auc(fpr, tpr)

    axes[1].plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.2f}"
    )

    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    axes[1].set_title(
        "Logistic Regression - ROC Curve",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")


    # =========================================================
    # 3. Decision Tree - Confusion Matrix
    # =========================================================

    y_pred_tree = tree_model.predict(X_test)

    cm_tree = confusion_matrix(
        y_test,
        y_pred_tree
    )

    sns.heatmap(
        cm_tree,
        annot=True,
        fmt="d",
        cmap="viridis",
        ax=axes[2]
    )

    axes[2].set_title(
        "Decision Tree - Confusion Matrix",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")


    # =========================================================
    # 4. Decision Tree - Structure
    # =========================================================

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        ax=axes[3]
    )

    axes[3].set_title(
        "Decision Tree - Structure",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    axes[3].axis("off")


    # =========================================================
    # 5. Decision Tree - Feature Importance
    # =========================================================

    importance = tree_model.feature_importances_

    axes[4].barh(
        feature_names,
        importance
    )

    axes[4].set_title(
        "Decision Tree - Feature Importance",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    axes[4].set_xlabel("Feature Importance")
    axes[4].set_ylabel("Features")


    # =========================================================
    # CLEAR SEPARATION BETWEEN VISUALIZATIONS
    # =========================================================

    plt.subplots_adjust(
        top=0.97,
        bottom=0.03,
        left=0.10,
        right=0.95,
        hspace=1.2
    )


    # =========================================================
    # SAVE ONE COMBINED IMAGE
    # =========================================================

    plt.savefig(
    "results/model_visualizations.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print("\nAll visualizations generated successfully!")
    print("Model visualizations saved at:")
print("results/model_visualizations.png")