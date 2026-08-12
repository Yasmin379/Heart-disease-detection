from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_test, predictions):

    #Calculate the main classification metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    #creating Confusion Matrix

    cm = confusion_matrix(y_test, predictions)

    #Storing results in a dictionary
    results = {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1_score":f1,
        "confusion_matrix":cm
    }

    return results

#Plotting Graphs for confusion matrix

def plot_confusion_matrix(cm, model_name,save_path):
    #Heatmap 

    # Create the folder if it does not already exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot = True,
        fmt = "d",
        cmap = "Blues",
        xticklabels = ["No Disease", "Disease"],
        yticklabels = ["No Disease", "Disease"]
    )

    plt.title(f"{model_name} - confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_comparison(results, save_path):
    """
    Creates a bar chart comparing the performance
    of different machine learning models.
    """
    # Create the folder if it does not already exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    models = list(results.keys())

    accuracy = [results[model]["accuracy"] * 100 for model in models]
    precision = [results[model]["precision"] * 100 for model in models]
    recall = [results[model]["recall"] * 100 for model in models]
    f1 = [results[model]["f1_score"] * 100 for model in models]

    x = range(len(models))
    width = 0.2

    plt.figure(figsize=(9, 5))

    plt.bar([i - 1.5 * width for i in x], accuracy,
            width, label="Accuracy")

    plt.bar([i - 0.5 * width for i in x], precision,
            width, label="Precision")

    plt.bar([i + 0.5 * width for i in x], recall,
            width, label="Recall")

    plt.bar([i + 1.5 * width for i in x], f1,
            width, label="F1 Score")

    plt.xticks(list(x), models)
    plt.ylabel("Score (%)")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()