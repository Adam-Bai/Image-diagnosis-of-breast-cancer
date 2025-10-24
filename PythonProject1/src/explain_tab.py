from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import joblib
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass
from src import load_wbcd, get_data_root

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="wbcd")
    ap.add_argument("--model_dir", type=str, default=None)
    ap.add_argument("--method", type=str, default="shap")
    ap.add_argument("--max_n", type=int, default=300)
    args = ap.parse_args()

    if args.task.lower() != "wbcd":
        raise SystemExit("Only wbcd supported in this minimal explainer.")
    X, y, feat = load_wbcd(get_data_root())

    if args.model_dir:
        mdir = Path(args.model_dir)
    else:
        outs = sorted(Path("outputs").glob("wbcd_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not outs:
            raise FileNotFoundError("No outputs/wbcd_* directory found. Train first.")
        mdir = outs[0]

    bundle = joblib.load(mdir / "model.pkl")
    model = bundle["model"]
    feature_names = bundle.get("feature_names", [f"f{i}" for i in range(X.shape[1])])

    n = min(args.max_n, X.shape[0])
    Xs = X[:n]

    import shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if hasattr(model, "predict_proba") and model.__class__.__name__.lower().startswith("randomforest"):
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(Xs)
        sv = vals[1] if isinstance(vals, list) else vals
    elif model.__class__.__name__.lower().startswith("logisticregression"):
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        explainer = shap.LinearExplainer(clf, Xs, feature_perturbation="interventional")
        sv = explainer.shap_values(Xs)
    else:
        explainer = shap.KernelExplainer(lambda data: model.predict_proba(data)[:,1], shap.sample(X, 100))
        sv = explainer.shap_values(Xs, nsamples=100)

    outdir = mdir / "shap"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sv, columns=feature_names).to_csv(outdir / "shap_values.csv", index=False)

    try:
        shap.summary_plot(sv, Xs, feature_names=feature_names, show=False)
        plt.tight_layout(); plt.savefig(outdir / "shap_summary.png", dpi=160); plt.close()
        shap.summary_plot(sv, Xs, feature_names=feature_names, show=False, plot_type="bar")
        plt.tight_layout(); plt.savefig(outdir / "shap_bar.png", dpi=160); plt.close()
    except Exception as e:
        print("Plotting failed:", e)

    print("Saved SHAP outputs to", outdir)

if __name__ == "__main__":
    main()
