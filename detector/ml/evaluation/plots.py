import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("results", exist_ok=True)


def save_training_plot(train_loss, val_loss):

    # -----------------------------
    # Loss Graph
    # -----------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        train_loss,
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        val_loss,
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig("results/loss.png")

    plt.close()


def save_accuracy_plot(train_accuracy, val_accuracy):

    # -----------------------------
    # Accuracy Graph
    # -----------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        train_accuracy,
        label="Training Accuracy",
        linewidth=2
    )

    plt.plot(
        val_accuracy,
        label="Validation Accuracy",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title("Training vs Validation Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig("results/accuracy.png")

    plt.close()


def save_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"]
    )

    fig, ax = plt.subplots(figsize=(6,6))

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    plt.title("Confusion Matrix")

    plt.savefig(
        "results/confusion_matrix.png"
    )

    plt.close()