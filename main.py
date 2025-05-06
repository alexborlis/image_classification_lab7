from dataset_loader import load_data
from classifier import train_knn_classifier, evaluate_classifier
import matplotlib.pyplot as plt
import numpy as np

X_train, X_test, y_train, y_test = load_data()
model = train_knn_classifier(X_train, y_train, k=5)
accuracy, predictions = evaluate_classifier(model, X_test, y_test)

# Save accuracy
with open("output/accuracy_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}")

# Visualization of some predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, image, pred, true in zip(axes.ravel(), X_test, predictions, y_test):
    ax.imshow(image.reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {pred}, True: {true}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("output/sample_predictions.png")
