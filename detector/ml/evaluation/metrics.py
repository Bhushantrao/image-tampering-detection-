from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "Fake",
            "Real"
        ],
        zero_division=0
    )

    print("\n")
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)

    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(report)