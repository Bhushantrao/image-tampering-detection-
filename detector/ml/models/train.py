# ============================================================
# Image Tampering Detection using ViT + ELA
# Research Version - Part 1
# ============================================================

import os
import time
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from sklearn.utils.class_weight import compute_class_weight

from .vit_model import ViTModel

from detector.ml.evaluation.metrics import evaluate

from detector.ml.evaluation.plots import (
    save_training_plot,
    save_confusion_matrix
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("="*60)
print("DEVICE INFORMATION")
print("="*60)

print("Using Device :", device)

if torch.cuda.is_available():

    print("GPU :", torch.cuda.get_device_name(0))

    print("CUDA :", torch.version.cuda)

else:

    print("Running on CPU")

# ============================================================
# PROJECT FOLDERS
# ============================================================

MODEL_FOLDER = "saved_models"

RESULT_FOLDER = "results"

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)

# ============================================================
# HYPER PARAMETERS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_EPOCHS = 25

LEARNING_RATE = 1e-4

NUM_CLASSES = 2

NUM_WORKERS = 2

PATIENCE = 5

MODEL_NAME = "ViT Base Patch16 224"

print("="*60)
print("TRAINING CONFIGURATION")
print("="*60)

print("Model :", MODEL_NAME)

print("Epochs :", NUM_EPOCHS)

print("Batch Size :", BATCH_SIZE)

print("Learning Rate :", LEARNING_RATE)

print("Image Size :", IMAGE_SIZE)

print("="*60)

# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

validation_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

print("Transforms Loaded Successfully")

# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DIR = "ela_dataset/train"

VAL_DIR = "ela_dataset/validation"

TEST_DIR = "ela_dataset/test"

print("\nLoading Dataset...")

# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=validation_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=validation_transform
)

# ============================================================
# DATASET INFORMATION
# ============================================================

print("="*60)
print("DATASET INFORMATION")
print("="*60)

print("Class Mapping :")
print(train_dataset.class_to_idx)

print()

print("Training Images :", len(train_dataset))

print("Validation Images :", len(val_dataset))

print("Testing Images :", len(test_dataset))

print()

print("Classes :", train_dataset.classes)

# ============================================================
# IMAGE COUNT
# ============================================================

fake_train = len(os.listdir(os.path.join(TRAIN_DIR, "fake")))
real_train = len(os.listdir(os.path.join(TRAIN_DIR, "real")))

fake_val = len(os.listdir(os.path.join(VAL_DIR, "fake")))
real_val = len(os.listdir(os.path.join(VAL_DIR, "real")))

fake_test = len(os.listdir(os.path.join(TEST_DIR, "fake")))
real_test = len(os.listdir(os.path.join(TEST_DIR, "real")))

print("\nTRAIN")

print("Fake :", fake_train)

print("Real :", real_train)

print("\nVALIDATION")

print("Fake :", fake_val)

print("Real :", real_val)

print("\nTEST")

print("Fake :", fake_test)

print("Real :", real_test)

# ============================================================
# CLASS WEIGHTS
# ============================================================

labels = train_dataset.targets

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)

print("\nClass Weights")

print(class_weights)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\nDataLoaders Created Successfully")

# ============================================================
# MODEL INITIALIZATION
# ============================================================

print("\n" + "="*60)
print("INITIALIZING MODEL")
print("="*60)

model = ViTModel().to(device)

print("Vision Transformer Loaded Successfully")

# ============================================================
# LOSS FUNCTION (Weighted Cross Entropy)
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

print("Weighted CrossEntropyLoss Initialized")

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

print("Optimizer : AdamW")

# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

print("Scheduler : ReduceLROnPlateau")

# ============================================================
# AUTOMATIC MIXED PRECISION (AMP)
# ============================================================

scaler = torch.cuda.amp.GradScaler(
    enabled=torch.cuda.is_available()
)

print("Mixed Precision :", torch.cuda.is_available())

# ============================================================
# TRAINING VARIABLES
# ============================================================

best_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)

early_stop_counter = 0

# ============================================================
# HISTORY
# ============================================================

train_losses = []

val_losses = []

train_accuracies = []

val_accuracies = []

learning_rates = []

print("\nTraining Variables Initialized")

# ============================================================
# MODEL INFORMATION
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)

print(f"Total Parameters      : {total_params:,}")
print(f"Trainable Parameters  : {trainable_params:,}")

with open(
    os.path.join(
        RESULT_FOLDER,
        "model_information.txt"
    ),
    "w"
) as f:

    f.write("Vision Transformer (ViT)\n\n")

    f.write(f"Model : {MODEL_NAME}\n")

    f.write(f"Total Parameters : {total_params:,}\n")

    f.write(f"Trainable Parameters : {trainable_params:,}\n")

    f.write(f"Epochs : {NUM_EPOCHS}\n")

    f.write(f"Batch Size : {BATCH_SIZE}\n")

    f.write(f"Learning Rate : {LEARNING_RATE}\n")

print("\nModel Information Saved")

# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

training_start_time = time.time()

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    epoch_start = time.time()

    # --------------------------
    # TRAIN MODE
    # --------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        # --------------------------
        # Mixed Precision
        # --------------------------

        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        if batch_idx % 20 == 0:

            print(
                f"Batch {batch_idx:4d}/{len(train_loader)} "
                f" Loss: {loss.item():.4f}"
            )

    # --------------------------
    # TRAIN RESULTS
    # --------------------------

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total

    train_losses.append(train_loss)

    train_accuracies.append(train_accuracy)

    print("\nTraining Results")

    print(f"Loss      : {train_loss:.4f}")

    print(f"Accuracy  : {train_accuracy:.2f}%")

    print(f"Time      : {time.time()-epoch_start:.1f} sec")
    
    # ============================================================
# VALIDATION LOOP
# ============================================================

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0

    y_true = []

    y_pred = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, predicted = torch.max(outputs,1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())

            y_pred.extend(predicted.cpu().numpy())

    # ------------------------------------

    val_loss = val_running_loss / len(val_loader)

    val_accuracy = 100 * val_correct / val_total

    val_losses.append(val_loss)

    val_accuracies.append(val_accuracy)

    learning_rates.append(
        optimizer.param_groups[0]["lr"]
    )

    scheduler.step(val_accuracy)

    print("\nValidation Results")

    print(f"Validation Loss     : {val_loss:.4f}")

    print(f"Validation Accuracy : {val_accuracy:.2f}%")

    # ============================================================
    # SAVE BEST MODEL
    # ============================================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(

            best_model_weights,

            os.path.join(
                MODEL_FOLDER,
                "best_vit_model.pth"
            )

        )

        print("\n✅ Best Model Saved")

        early_stop_counter = 0

    else:

        early_stop_counter += 1

        print(

            f"No Improvement ({early_stop_counter}/{PATIENCE})"

        )

    # ============================================================
    # EARLY STOPPING
    # ============================================================

    if early_stop_counter >= PATIENCE:

        print("\nEarly Stopping Triggered")

        break

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "="*60)
print("TESTING BEST MODEL")
print("="*60)

model.load_state_dict(best_model_weights)

model.eval()

test_loss = 0.0

test_correct = 0

test_total = 0

y_true = []

y_pred = []

probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)

        test_correct += (predicted == labels).sum().item()

        y_true.extend(labels.cpu().numpy())

        y_pred.extend(predicted.cpu().numpy())

        probabilities.extend(
            probs[:,1].cpu().numpy()
        )

# ============================================================
# FINAL TEST RESULTS
# ============================================================

test_loss = test_loss / len(test_loader)

test_accuracy = 100 * test_correct / test_total

print("\nFinal Test Results")

print(f"Test Loss     : {test_loss:.4f}")

print(f"Test Accuracy : {test_accuracy:.2f}%")

# ============================================================
# CALCULATE METRICS
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

accuracy = accuracy_score(y_true,y_pred)

precision = precision_score(y_true,y_pred)

recall = recall_score(y_true,y_pred)

f1 = f1_score(y_true,y_pred)

cm = confusion_matrix(y_true,y_pred)

print("\n" + "="*60)

print("FINAL METRICS")

print("="*60)

print(f"Accuracy  : {accuracy*100:.2f}%")

print(f"Precision : {precision*100:.2f}%")

print(f"Recall    : {recall*100:.2f}%")

print(f"F1 Score  : {f1*100:.2f}%")

print("\nConfusion Matrix")

print(cm)

print("\nClassification Report")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Fake","Real"]
    )
)

# ============================================================
# SAVE METRICS
# ============================================================

with open(
    os.path.join(
        RESULT_FOLDER,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(
        classification_report(
            y_true,
            y_pred,
            target_names=["Fake","Real"]
        )
    )

print("\nClassification Report Saved")

# ============================================================
# PART 7 : SAVE RESULTS & GENERATE REPORTS
# ============================================================

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

print("\nGenerating Research Paper Results...")

# ============================================================
# TRAINING TIME
# ============================================================

training_end_time = time.time()

training_time = training_end_time - training_start_time

print(f"Training Time : {training_time:.2f} seconds")

# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Training Loss", linewidth=2)

plt.plot(val_losses, label="Validation Loss", linewidth=2)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training & Validation Loss")

plt.grid(True)

plt.legend()

plt.savefig(
    os.path.join(RESULT_FOLDER,"loss_curve.png"),
    dpi=300
)

plt.close()

# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(train_accuracies,label="Training Accuracy",linewidth=2)

plt.plot(val_accuracies,label="Validation Accuracy",linewidth=2)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title("Training & Validation Accuracy")

plt.grid(True)

plt.legend()

plt.savefig(
    os.path.join(RESULT_FOLDER,"accuracy_curve.png"),
    dpi=300
)

plt.close()

# ============================================================
# CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake","Real"]
)

fig, ax = plt.subplots(figsize=(6,6))

disp.plot(
    cmap="Blues",
    ax=ax
)

plt.title("Confusion Matrix")

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# ROC CURVE
# ============================================================

fpr,tpr,_ = roc_curve(
    y_true,
    probabilities
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.grid(True)

plt.legend()

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "roc_curve.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# PRECISION RECALL CURVE
# ============================================================

precision_curve,recall_curve,_ = precision_recall_curve(
    y_true,
    probabilities
)

plt.figure(figsize=(7,6))

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision Recall Curve")

plt.grid(True)

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "precision_recall_curve.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# EXCEL REPORT
# ============================================================

report = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "Training Time (sec)"

    ],

    "Value":[

        accuracy*100,

        precision*100,

        recall*100,

        f1*100,

        training_time

    ]

})

report.to_excel(

    os.path.join(
        RESULT_FOLDER,
        "results.xlsx"
    ),

    index=False

)

# ============================================================
# TRAINING LOG
# ============================================================

log = pd.DataFrame({

    "Epoch":list(range(1,len(train_losses)+1)),

    "Training Loss":train_losses,

    "Validation Loss":val_losses,

    "Training Accuracy":train_accuracies,

    "Validation Accuracy":val_accuracies,

    "Learning Rate":learning_rates

})

log.to_csv(

    os.path.join(
        RESULT_FOLDER,
        "training_log.csv"
    ),

    index=False

)

# ============================================================
# GPU INFORMATION
# ============================================================

with open(

    os.path.join(
        RESULT_FOLDER,
        "gpu_information.txt"
    ),

    "w"

) as f:

    if torch.cuda.is_available():

        f.write(

            f"GPU : {torch.cuda.get_device_name(0)}\n"

        )

        f.write(

            f"CUDA Version : {torch.version.cuda}\n"

        )

    else:

        f.write("CPU\n")

print("="*60)

print("ALL RESULTS GENERATED SUCCESSFULLY")

print("="*60)

print("Saved Files")

print("---------------------------")

print("loss_curve.png")

print("accuracy_curve.png")

print("roc_curve.png")

print("precision_recall_curve.png")

print("confusion_matrix.png")

print("classification_report.txt")

print("results.xlsx")

print("training_log.csv")

print("gpu_information.txt")

print("model_information.txt")

print("\nResearch Paper Results Ready")

