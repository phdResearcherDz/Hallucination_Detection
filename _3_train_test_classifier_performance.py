#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PART 3 — MedHallu Evaluation Protocol (Enhanced)

This script:
 - Loads features from PART 2 (features.jsonl)
 - Reconstructs train/validation/test splits
 - Trains:
       * Binary hallucination detector
       * Multi-class hallucination category classifier
 - Evaluates using MedHallu-style metrics
 - Outputs:
       * Accuracy, Precision, Recall, F1
       * Macro-F1
       * Per-class classification report
       * Confusion matrix
       * Feature importance (top-K)
"""

import os
import json
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


###############################################################################
# CONFIG
###############################################################################

FEATURES_PATH = "features_not_labeled/features.jsonl"
TOP_K_FEATURES = 30     # export top feature importance


###############################################################################
# LOADING FEATURES
###############################################################################

def load_features(path):
    print("\n=== Loading features from:", path, "===")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature file not found: {path}\nRun PART 2 first."
        )

    X_list, y_bin_list, y_cat_list, split_list = [], [], [], []
    feature_keys = None

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except:
                print(f"⚠️ Skipping malformed JSON on line {line_num}")
                continue

            if "binary_label" not in obj or "category" not in obj or "split" not in obj:
                print(f"⚠️ Missing fields in line {line_num}, skipping.")
                continue

            # determine feature keys once
            if feature_keys is None:
                excluded = {"binary_label", "category", "split"}
                feature_keys = sorted([k for k in obj if k not in excluded])
                print("\nDetected", len(feature_keys), "feature dimensions.")
                print("Sample keys:", feature_keys[:10], "...\n")

            X_list.append([float(obj[k]) for k in feature_keys])
            y_bin_list.append(int(obj["binary_label"]))
            y_cat_list.append(obj["category"])
            split_list.append(obj["split"])

    X = np.array(X_list, dtype=np.float32)
    y_bin = np.array(y_bin_list, dtype=np.int64)
    y_cat = np.array(y_cat_list, dtype=object)
    splits = np.array(split_list, dtype=object)

    print(f"✓ Loaded {len(X)} instances with {X.shape[1]} features.\n")

    return X, y_bin, y_cat, splits, feature_keys


###############################################################################
# BINARY SPLITTING
###############################################################################

def build_binary_splits(X, y_bin, splits):
    print("=== Building BINARY classification splits ===")

    mask_train = splits == "train"
    mask_val   = splits == "validation"
    mask_test  = splits == "test"

    Xtr, ytr = X[mask_train], y_bin[mask_train]
    Xv,  yv  = X[mask_val], y_bin[mask_val]
    Xte, yte = X[mask_test], y_bin[mask_test]

    print(f" Train: {len(Xtr)}, Val: {len(Xv)}, Test: {len(Xte)}")

    return Xtr, ytr, Xv, yv, Xte, yte


###############################################################################
# MULTI-CLASS SPLITTING (HALLUCINATED ONLY)
###############################################################################

def build_multiclass_splits(X, y_cat, y_bin, splits):
    print("\n=== Building MULTI-CLASS splits ===")

    # Only keep hallucinated answers (binary_label=1)
    mask_h = (y_bin == 1)
    X_h = X[mask_h]
    y_h = y_cat[mask_h]
    splits_h = splits[mask_h]

    # filter out 'none'
    mask_valid = (y_h != "none")
    X_h = X_h[mask_valid]
    y_h = y_h[mask_valid]
    splits_h = splits_h[mask_valid]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_h)

    print(" Discovered categories:", list(le.classes_))

    Xtr = X_h[splits_h == "train"]
    ytr = y_enc[splits_h == "train"]

    Xv  = X_h[splits_h == "validation"]
    yv  = y_enc[splits_h == "validation"]

    Xte = X_h[splits_h == "test"]
    yte = y_enc[splits_h == "test"]

    print(f" Train: {len(Xtr)}, Val: {len(Xv)}, Test: {len(Xte)}")

    return Xtr, ytr, Xv, yv, Xte, yte, le


###############################################################################
# MODEL TRAINING
###############################################################################

def train_binary_model(Xtr, ytr):
    print("\n=== Training BINARY XGBoost ===")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(Xtr, ytr)
    return model


def train_multiclass_model(Xtr, ytr, num_classes):
    print("\n=== Training MULTI-CLASS XGBoost ===")
    model = XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(Xtr, ytr)
    return model


###############################################################################
# EVALUATION
###############################################################################

def eval_binary(model, Xte, yte):
    print("\n=== BINARY HALLUCINATION DETECTION RESULTS ===")
    y_pred = model.predict(Xte)

    acc = accuracy_score(yte, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        yte, y_pred, average="binary", zero_division=0
    )

    print(f" Accuracy : {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall   : {rec:.4f}")
    print(f" F1       : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(yte, y_pred))


def eval_multiclass(model, Xte, yte, le):
    print("\n=== MULTI-CLASS HALLUCINATION CATEGORY RESULTS ===")
    y_pred = model.predict(Xte)

    labels = list(range(len(le.classes_)))   # Always expect ALL classes

    print("Expected labels:", labels)
    print("Classes:", le.classes_)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        yte, y_pred,
        average="macro",
        zero_division=0,
        labels=labels                      # <-- NEW
    )

    print(f" Macro Precision: {macro_p:.4f}")
    print(f" Macro Recall   : {macro_r:.4f}")
    print(f" Macro F1       : {macro_f1:.4f}\n")

    print("Per-class report:")
    print(classification_report(
        yte,
        y_pred,
        target_names=le.classes_,
        labels=labels,                    # <-- NEW
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(le.classes_)
    print(confusion_matrix(yte, y_pred, labels=labels))

###############################################################################
# FEATURE IMPORTANCE EXPORT
###############################################################################

def export_feature_importance(model, feature_keys, out_path="features/top_features.txt"):
    print("\n=== Exporting Feature Importances ===")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    importance = model.feature_importances_

    # sort by importance
    idx = np.argsort(importance)[::-1]
    top = idx[:TOP_K_FEATURES]

    with open(out_path, "w", encoding="utf-8") as f:
        for i in top:
            f.write(f"{feature_keys[i]} : {importance[i]:.6f}\n")

    print(f"✓ Saved top-{TOP_K_FEATURES} features → {out_path}")


###############################################################################
# MAIN
###############################################################################

def main():
    # ---- Load features ----
    X, y_bin, y_cat, splits, feature_keys = load_features(FEATURES_PATH)

    # ---- Binary classification ----
    Xtr, ytr, Xv, yv, Xte, yte = build_binary_splits(X, y_bin, splits)
    model_bin = train_binary_model(Xtr, ytr)
    eval_binary(model_bin, Xte, yte)
    export_feature_importance(model_bin, feature_keys,
                              out_path="features/binary_top_features.txt")

    # ---- Multi-class classification ----
    Xmtr, ymtr, Xmval, ymval, Xmte, ymte, le = build_multiclass_splits(
        X, y_cat, y_bin, splits
    )
    model_mc = train_multiclass_model(Xmtr, ymtr, len(le.classes_))
    eval_multiclass(model_mc, Xmte, ymte, le)
    export_feature_importance(model_mc, feature_keys,
                              out_path="features/multiclass_top_features.txt")


    print("\n=== PART 3 COMPLETE ===")


if __name__ == "__main__":
    main()
