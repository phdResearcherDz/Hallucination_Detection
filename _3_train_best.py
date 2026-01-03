#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

FEATURES_PATH = "features/features.jsonl"

def load_features(path):
    X, y_bin, y_cat, splits = [], [], [], []
    feature_keys = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            if feature_keys is None:
                excluded = {"binary_label", "category", "split"}
                feature_keys = sorted([k for k in obj if k not in excluded])

            X.append([obj[k] for k in feature_keys])
            y_bin.append(obj["binary_label"])
            y_cat.append(obj["category"])
            splits.append(obj["split"])

    return np.array(X, np.float32), np.array(y_bin), np.array(y_cat), np.array(splits), feature_keys


def main():
    X, y_bin, y_cat, splits, feature_keys = load_features(FEATURES_PATH)

    # --- BINARY CLASSIFICATION ---
    mask_train = splits == "train"
    mask_val   = splits == "validation"
    mask_test  = splits == "test"

    Xtr, ytr = X[mask_train], y_bin[mask_train]
    Xv,  yv  = X[mask_val], y_bin[mask_val]
    Xte, yte = X[mask_test], y_bin[mask_test]

    # class weights fix imbalance
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    class_weights = {i: cw[i] for i in range(len(cw))}

    print("\n=== Training TABNET (Binary) ===")

    model_bin = TabNetClassifier(
        n_d=32, n_a=32,
        n_steps=5,
        gamma=1.3,
        momentum=0.02,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
    )

    model_bin.fit(
        Xtr, ytr,
        eval_set=[(Xv, yv)],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=100,
        patience=10,
        batch_size=32,
        virtual_batch_size=16,
        num_workers=0,
        weights=class_weights
    )

    preds = model_bin.predict(Xte)
    print("\nBinary accuracy:", accuracy_score(yte, preds))
    print(confusion_matrix(yte, preds))

    # --- MULTI-CLASS ---
    print("\n=== Training TABNET (Multi-class) ===")

    mask = (y_bin == 1) & (y_cat != "none")
    X_h = X[mask]
    y_h = y_cat[mask]
    splits_h = splits[mask]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_h)

    Xtr = X_h[splits_h == "train"]
    ytr = y_enc[splits_h == "train"]

    Xv = X_h[splits_h == "validation"]
    yv = y_enc[splits_h == "validation"]

    Xte = X_h[splits_h == "test"]
    yte = y_enc[splits_h == "test"]

    model_mc = TabNetClassifier(
        n_d=32, n_a=32,
        n_steps=5,
        gamma=1.3,
        momentum=0.02,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
    )

    model_mc.fit(
        Xtr, ytr,
        eval_set=[(Xv, yv)],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=200,
        patience=20,
        batch_size=32,
        virtual_batch_size=16,
    )

    preds = model_mc.predict(Xte)

    print("\nMulti-class Report:")
    print(classification_report(yte, preds, target_names=le.classes_))
    print(confusion_matrix(yte, preds))


if __name__ == "__main__":
    main()
