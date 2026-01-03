import os
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------
# OPTIONAL DEPENDENCIES
# ---------------------------------------------------------------------
XGB_AVAILABLE = False
LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
STAGE_A_PATH = "features_not_labeled/features.jsonl"   # 90% train + 10% validation
STAGE_B_PATH = "features_labeled/features.jsonl"       # labeled test

OUT_DIR = "results_binary_ v5"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Torch models
TAB_BATCH_SIZE = 64
TAB_EPOCHS = 80
TAB_LR = 2e-4

# FeatBERT (Transformer over features)
FB_D_MODEL = 128
FB_NHEAD = 4
FB_NLAYERS = 4
FB_DROPOUT = 0.15
FB_EPOCHS = 40
FB_LR = 3e-4

# Threshold tuning grid
THRESH_GRID = np.linspace(0.05, 0.95, 19)


# ---------------------------------------------------------------------
# FEATURE GROUP DEFINITIONS (by key)
# ---------------------------------------------------------------------
TEXT_KEYS = {"sim_a_q", "sim_a_k", "jaccard_a_q", "jaccard_a_k"}
GRAPH_LEVEL_KEYS = {"graph_sim_a_q", "graph_sim_a_k", "nodes_a", "edges_a", "density_a", "coherence_a"}

def is_alignment_key(k: str) -> bool:
    return k.startswith("Q_") or k.startswith("K_")


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find file: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows

def extract_feature_keys(rows: List[dict]) -> List[str]:
    # ignore labels + split + possible raw text keys
    exclude = {"binary_label", "category", "split", "question", "knowledge", "answer"}
    return sorted([k for k in rows[0].keys() if k not in exclude])

def build_matrix(rows: List[dict], feature_keys: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([[safe_float(r.get(k, 0.0)) for k in feature_keys] for r in rows], dtype=np.float32)
    y = np.array([int(r["binary_label"]) for r in rows], dtype=np.int64)
    return X, y

def binary_metrics(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"Accuracy": float(acc), "Precision": float(prec), "Recall": float(rec), "F1": float(f1)}

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def tune_threshold(y_true: np.ndarray, scores: np.ndarray, grid=THRESH_GRID) -> Tuple[float, Dict[str, float]]:
    best_t = 0.5
    best_m = {"F1": -1.0, "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0}
    for t in grid:
        pred = (scores >= t).astype(int)
        m = binary_metrics(y_true, pred)
        if m["F1"] > best_m["F1"]:
            best_m = m
            best_t = float(t)
    return best_t, best_m


# ---------------------------------------------------------------------
# FIGURES (PDF)
# ---------------------------------------------------------------------
def save_bar_pdf(df: pd.DataFrame, value_col: str, outpath: str, ymin_pad: float = 0.02):
    """
    No title. Y axis zoom around min/max.
    """
    plt.figure()
    x = np.arange(len(df))
    plt.bar(x, df[value_col].values)
    plt.xticks(x, df["Run"].values, rotation=35, ha="right")
    plt.ylabel(value_col)

    vals = df[value_col].values.astype(float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if math.isfinite(vmin) and math.isfinite(vmax) and vmax > vmin:
        span = vmax - vmin
        ymin = max(0.0, vmin - ymin_pad * max(span, 1e-6))
        ymax = min(1.0, vmax + ymin_pad * max(span, 1e-6))
        if ymax - ymin < 0.05:
            mid = (ymax + ymin) / 2
            ymin = max(0.0, mid - 0.03)
            ymax = min(1.0, mid + 0.03)
        plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()

def save_confusion_pdf(cm: np.ndarray, labels: List[str], outpath: str):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# FEATURE SUBSETS
# ---------------------------------------------------------------------
def build_feature_subsets(feature_keys: List[str]) -> Dict[str, List[int]]:
    idx_text = [i for i, k in enumerate(feature_keys) if k in TEXT_KEYS]
    idx_graph_level = [i for i, k in enumerate(feature_keys) if k in GRAPH_LEVEL_KEYS]
    idx_align = [i for i, k in enumerate(feature_keys) if is_alignment_key(k)]
    idx_all = list(range(len(feature_keys)))
    return {
        "TextOnly": idx_text,
        "GraphLevelOnly": idx_graph_level,
        "AlignmentOnly": idx_align,
        "AllFeatures": idx_all,
    }


# ---------------------------------------------------------------------
# BASELINE: RANDOM
# ---------------------------------------------------------------------
def run_random(y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    preds = (np.random.rand(len(y_test)) < 0.5).astype(int)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    return binary_metrics(y_test, preds), cm


# ---------------------------------------------------------------------
# SKLEARN MODELS (return probabilities for tuning)
# ---------------------------------------------------------------------
def fit_predict_scores_sklearn(model_name: str,
                               Xtr: np.ndarray, ytr: np.ndarray,
                               Xva: np.ndarray, yva: np.ndarray,
                               Xte: np.ndarray) -> np.ndarray:
    """
    Returns scores in [0,1] for the positive class.
    """
    if model_name == "LogReg":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
        ])

    elif model_name == "RF":
        model = RandomForestClassifier(
            n_estimators=1200, random_state=SEED, class_weight="balanced", n_jobs=-1,
            max_features="sqrt"
        )

    elif model_name == "ExtraTrees":
        model = ExtraTreesClassifier(
            n_estimators=1600, random_state=SEED, class_weight="balanced", n_jobs=-1,
            max_features="sqrt"
        )

    elif model_name == "GradBoost":
        model = GradientBoostingClassifier(random_state=SEED)

    elif model_name == "AdaBoost":
        model = AdaBoostClassifier(random_state=SEED, n_estimators=600, learning_rate=0.5)

    elif model_name == "SVM-RBF":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(C=2.0, kernel="rbf", gamma="scale", class_weight="balanced"))
        ])
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    elif model_name == "SkMLP":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=400,
                random_state=SEED,
                early_stopping=True,
                n_iter_no_change=15
            ))
        ])

    else:
        raise ValueError(model_name)

    model.fit(Xtr, ytr)

    # prefer calibrated probabilities
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(Xte)[:, 1]
        return scores.astype(np.float32)

    # fallback for decision_function
    if hasattr(model, "decision_function"):
        df = model.decision_function(Xte)
        return sigmoid(df).astype(np.float32)

    # final fallback
    preds = model.predict(Xte).astype(int)
    return preds.astype(np.float32)


# ---------------------------------------------------------------------
# XGBOOST / LIGHTGBM
# ---------------------------------------------------------------------
def fit_predict_scores_xgb(Xtr, ytr, Xva, yva, Xte) -> np.ndarray:
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost not installed")

    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        verbose=False
    )
    scores = model.predict_proba(Xte)[:, 1]
    return scores.astype(np.float32)

def fit_predict_scores_lgbm(Xtr, ytr, Xva, yva, Xte) -> np.ndarray:
    if not LGBM_AVAILABLE:
        raise RuntimeError("lightgbm not installed")

    model = lgb.LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="binary_logloss"
    )
    scores = model.predict_proba(Xte)[:, 1]
    return scores.astype(np.float32)


# ---------------------------------------------------------------------
# TORCH MODEL 1: TabMLP
# ---------------------------------------------------------------------
class TabMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.10),

            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def make_loader(X: np.ndarray, y: Optional[np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    if y is None:
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    else:
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    bs = min(batch_size, len(ds)) if len(ds) > 0 else batch_size
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

def train_torch(model: nn.Module,
                Xtr: np.ndarray, ytr: np.ndarray,
                Xva: np.ndarray, yva: np.ndarray,
                epochs: int, lr: float, batch_size: int) -> nn.Module:
    tr_loader = make_loader(Xtr, ytr, batch_size, shuffle=True)
    va_loader = make_loader(Xva, yva, batch_size, shuffle=False)

    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    patience, wait = 10, 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl += float(crit(model(xb), yb).item())
        vl /= max(1, len(va_loader))

        if vl < best_val:
            best_val = vl
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def torch_scores(model: nn.Module, X: np.ndarray) -> np.ndarray:
    loader = make_loader(X, None, batch_size=TAB_BATCH_SIZE, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            p = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probs.extend(list(p))
    return np.array(probs, dtype=np.float32)


# ---------------------------------------------------------------------
# TORCH MODEL 2: FeatBERT — Transformer encoder over feature tokens
# ---------------------------------------------------------------------
class FeatBERT(nn.Module):
    """
    "BERT-style" feature transformer:
      - Treat each scalar feature as a token
      - Embed token id + scalar value -> d_model
      - TransformerEncoder -> pooled representation -> classifier
    """
    def __init__(self, num_feats: int, d_model: int = 128, nhead: int = 4, nlayers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.id_emb = nn.Embedding(num_feats, d_model)
        self.val_proj = nn.Linear(1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 2)

        self.register_buffer("feat_ids", torch.arange(num_feats, dtype=torch.long).unsqueeze(0), persistent=False)

    def forward(self, x):
        B, D = x.shape
        ids = self.feat_ids.expand(B, D)    # [B, D]
        id_e = self.id_emb(ids)             # [B, D, d_model]
        v_e = self.val_proj(x.unsqueeze(-1))# [B, D, d_model]
        h = id_e + v_e
        h = self.encoder(h)
        h = h.mean(dim=1)
        h = self.dropout(h)
        return self.head(h)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    set_seed(SEED)
    ensure_dirs()

    # ----------------------------
    # Load Stage A and Stage B
    # ----------------------------
    rows_a = load_jsonl(STAGE_A_PATH)
    rows_b = load_jsonl(STAGE_B_PATH)

    # Use Stage A keys as canonical; enforce Stage B compatibility
    feature_keys = extract_feature_keys(rows_a)
    feature_keys_b = extract_feature_keys(rows_b)
    if feature_keys_b != feature_keys:
        missing = set(feature_keys) - set(feature_keys_b)
        extra = set(feature_keys_b) - set(feature_keys)
        raise ValueError(
            "Feature mismatch between Stage A and Stage B.\n"
            f"Missing in Stage B: {sorted(list(missing))[:30]}\n"
            f"Extra in Stage B: {sorted(list(extra))[:30]}"
        )

    Xa, ya = build_matrix(rows_a, feature_keys)
    Xb, yb = build_matrix(rows_b, feature_keys)

    # Stage A has train/validation splits already
    splits_a = np.array([r.get("split", "train") for r in rows_a], dtype=object)
    tr_mask = (splits_a == "train")
    va_mask = (splits_a == "validation")
    if tr_mask.sum() == 0 or va_mask.sum() == 0:
        raise ValueError("Stage A must contain split labels with both 'train' and 'validation'.")

    subsets = build_feature_subsets(feature_keys)

    # models list
    sklearn_models = ["LogReg", "RF", "ExtraTrees", "GradBoost", "AdaBoost", "SVM-RBF", "SkMLP"]
    if XGB_AVAILABLE:
        sklearn_models.append("XGBoost")
    if LGBM_AVAILABLE:
        sklearn_models.append("LightGBM")

    runs = []
    best_overall = {"F1": -1.0, "cm": None, "run": None, "subset": None}

    # ----------------------------
    # Random baseline (evaluated on Stage B only)
    # ----------------------------
    m_rand, cm_rand = run_random(yb)
    runs.append({
        "Subset": "AllFeatures",
        "Run": "Random",
        "Threshold": 0.50,
        **m_rand
    })
    if m_rand["F1"] > best_overall["F1"]:
        best_overall.update({"F1": m_rand["F1"], "cm": cm_rand, "run": "Random", "subset": "AllFeatures"})

    # ----------------------------
    # Evaluate each subset
    # ----------------------------
    for subset_name, idxs in subsets.items():
        if len(idxs) == 0:
            print(f"[WARN] Subset {subset_name} is empty; skipping.")
            continue

        Xtr = Xa[tr_mask][:, idxs]
        ytr = ya[tr_mask]
        Xva = Xa[va_mask][:, idxs]
        yva = ya[va_mask]
        Xte = Xb[:, idxs]
        yte = yb

        # Standardization used for Torch + some sklearn pipelines already handle scaling
        scaler = StandardScaler()
        scaler.fit(Xtr)
        Xtr_std = scaler.transform(Xtr).astype(np.float32)
        Xva_std = scaler.transform(Xva).astype(np.float32)
        Xte_std = scaler.transform(Xte).astype(np.float32)

        print(f"\n=== SUBSET: {subset_name} | dim={len(idxs)} ===")

        # ---- SKLEARN / BOOSTING ----
        for mdl in sklearn_models:
            if mdl == "XGBoost":
                scores_va = fit_predict_scores_xgb(Xtr, ytr, Xva, yva, Xva)
                scores_te = fit_predict_scores_xgb(Xtr, ytr, Xva, yva, Xte)
            elif mdl == "LightGBM":
                scores_va = fit_predict_scores_lgbm(Xtr, ytr, Xva, yva, Xva)
                scores_te = fit_predict_scores_lgbm(Xtr, ytr, Xva, yva, Xte)
            else:
                scores_va = fit_predict_scores_sklearn(mdl, Xtr, ytr, Xva, yva, Xva)
                scores_te = fit_predict_scores_sklearn(mdl, Xtr, ytr, Xva, yva, Xte)

            thr, _ = tune_threshold(yva, scores_va)
            preds = (scores_te >= thr).astype(int)
            m = binary_metrics(yte, preds)
            cm = confusion_matrix(yte, preds, labels=[0, 1])

            run_name = f"{mdl}-{subset_name}"
            runs.append({"Subset": subset_name, "Run": run_name, "Threshold": thr, **m})

            if m["F1"] > best_overall["F1"]:
                best_overall.update({"F1": m["F1"], "cm": cm, "run": run_name, "subset": subset_name})

        # ---- TORCH: TabMLP ----
        tabmlp = TabMLP(input_dim=Xtr_std.shape[1], output_dim=2)
        tabmlp = train_torch(tabmlp, Xtr_std, ytr, Xva_std, yva, epochs=TAB_EPOCHS, lr=TAB_LR, batch_size=TAB_BATCH_SIZE)
        scores_va = torch_scores(tabmlp, Xva_std)
        scores_te = torch_scores(tabmlp, Xte_std)
        thr, _ = tune_threshold(yva, scores_va)
        preds = (scores_te >= thr).astype(int)
        m = binary_metrics(yte, preds)
        cm = confusion_matrix(yte, preds, labels=[0, 1])

        run_name = f"TabMLP-{subset_name}"
        runs.append({"Subset": subset_name, "Run": run_name, "Threshold": thr, **m})
        if m["F1"] > best_overall["F1"]:
            best_overall.update({"F1": m["F1"], "cm": cm, "run": run_name, "subset": subset_name})

        # ---- TORCH: FeatBERT ----
        featbert = FeatBERT(
            num_feats=Xtr_std.shape[1],
            d_model=FB_D_MODEL,
            nhead=FB_NHEAD,
            nlayers=FB_NLAYERS,
            dropout=FB_DROPOUT
        )
        featbert = train_torch(featbert, Xtr_std, ytr, Xva_std, yva, epochs=FB_EPOCHS, lr=FB_LR, batch_size=TAB_BATCH_SIZE)
        scores_va = torch_scores(featbert, Xva_std)
        scores_te = torch_scores(featbert, Xte_std)
        thr, _ = tune_threshold(yva, scores_va)
        preds = (scores_te >= thr).astype(int)
        m = binary_metrics(yte, preds)
        cm = confusion_matrix(yte, preds, labels=[0, 1])

        run_name = f"FeatBERT-{subset_name}"
        runs.append({"Subset": subset_name, "Run": run_name, "Threshold": thr, **m})
        if m["F1"] > best_overall["F1"]:
            best_overall.update({"F1": m["F1"], "cm": cm, "run": run_name, "subset": subset_name})

    # ----------------------------
    # Export CSV
    # ----------------------------
    df = pd.DataFrame(runs).sort_values(["Subset", "F1"], ascending=[True, False])
    out_csv = os.path.join(TABLE_DIR, "binary_results.csv")
    df.to_csv(out_csv, index=False)

    # ----------------------------
    # 4 bar charts (one per subset)
    # ----------------------------
    for subset_name in ["TextOnly", "GraphLevelOnly", "AlignmentOnly", "AllFeatures"]:
        dsub = df[df["Subset"] == subset_name].copy()
        if dsub.empty:
            continue
        # Sort by F1 and keep all runs (you can .head(15) if too crowded)
        dsub = dsub.sort_values("F1", ascending=False)
        out_bar = os.path.join(FIG_DIR, f"binary_f1_bar_{subset_name}.pdf")
        save_bar_pdf(dsub, "F1", out_bar)

    # ----------------------------
    # Best confusion matrix overall
    # ----------------------------
    out_cm = os.path.join(FIG_DIR, "binary_confusion_best.pdf")
    if best_overall["cm"] is not None:
        save_confusion_pdf(best_overall["cm"], ["0", "1"], out_cm)

    print("\n=== DONE (BINARY, STAGE A/B) ===")
    print("CSV:", out_csv)
    print("Best overall:", best_overall["run"], "| subset:", best_overall["subset"], "| F1:", best_overall["F1"])
    print("Figures:", FIG_DIR)
    if not XGB_AVAILABLE:
        print("[INFO] xgboost not available; install via: pip install xgboost")
    if not LGBM_AVAILABLE:
        print("[INFO] lightgbm not available; install via: pip install lightgbm")

if __name__ == "__main__":
    main()
