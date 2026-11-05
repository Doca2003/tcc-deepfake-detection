import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_ds, threshold=0.5):
    y_true, y_prob = [], []
    for images, labels in test_ds:
        y_prob.extend(model.predict(images, verbose=0).ravel())
        y_true.extend(labels.numpy())

    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_prob) >= threshold).astype(int)

    return{
        "accurancy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true,y_pred,zero_division=0)),
        "recall": float(recall_score(y_true,y_pred,zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }