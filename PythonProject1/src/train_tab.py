from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
from src import load_wbcd, get_data_root

def build_model(name: str):
    name = name.lower()
    if name in ("logreg","lr","logistic"):
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, solver="lbfgs"))])
    elif name in ("rf","randomforest"):
        return RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=0, class_weight="balanced_subsample")
    elif name in ("xgb","xgboost"):
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss", n_jobs=-1)
        except Exception as e:
            print("xgboost not available, falling back to RandomForest. Reason:", e)
            return RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=0, class_weight="balanced_subsample")
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="wbcd")
    ap.add_argument("--model", type=str, default="logreg", help="logreg | rf | xgboost")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    if args.task.lower() != "wbcd":
        raise SystemExit("This minimal runner currently supports --task wbcd only.")
    X, y, feature_names = load_wbcd(get_data_root())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    aucs, prs, f1s = [], [], []
    oof_pred = np.zeros_like(y, dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        mdl = build_model(args.model)
        mdl.fit(X[tr], y[tr])
        proba = mdl.predict_proba(X[va])[:,1] if hasattr(mdl, "predict_proba") else mdl.decision_function(X[va])
        aucs.append(roc_auc_score(y[va], proba))
        prs.append(average_precision_score(y[va], proba))
        f1s.append(f1_score(y[va], (proba>=0.5).astype(int)))
        oof_pred[va] = proba

    outdir = Path(args.outdir) if args.outdir else Path("outputs") / f"wbcd_{args.model}_s{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    final_model = build_model(args.model)
    final_model.fit(X, y)
    joblib.dump({"model": final_model, "feature_names": feature_names}, outdir / "model.pkl")

    metrics = {
        "cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
        "cv_pr_mean": float(np.mean(prs)), "cv_pr_std": float(np.std(prs)),
        "cv_f1_mean": float(np.mean(f1s)), "cv_f1_std": float(np.std(f1s)),
        "fold_auc": [float(a) for a in aucs]
    }
    import json
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved to:", outdir)
    print(metrics)

if __name__ == "__main__":
    main()
