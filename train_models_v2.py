# -*- coding: utf-8 -*-
"""
训练脚本 v2：用 03_credit_default_features.csv（含 6 个业务衍生特征）训练多模型，
并把“调参过程”与“模型对比”落成可引用的产物，供界面（app.py）与 PPT 使用。

运行：python train_models_v2.py
产物（models_v2/）：
    lr.pkl / rf_tuned.pkl / gbdt.pkl
    metrics_v2.json            各模型验证集 / 测试集指标、阈值
    tuning_log.json            搜索空间、每一组参数得分、最优参数
    tuning_before_after.csv    调参前后对比表
    model_comparison_v2.csv    三个模型横向对比表
    feature_list.json          入模特征清单（基础 23 + 衍生 6）
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CSV = BASE_DIR / "data" / "03_credit_default_features.csv"
OUT = BASE_DIR / "models_v2"
OUT.mkdir(exist_ok=True)

BASE_FEATURES = [
    "credit_limit_ntd", "sex_code", "education_code", "marriage_code", "age_years",
    "pay_status_2005_09", "pay_status_2005_08", "pay_status_2005_07",
    "pay_status_2005_06", "pay_status_2005_05", "pay_status_2005_04",
    "bill_amount_2005_09_ntd", "bill_amount_2005_08_ntd", "bill_amount_2005_07_ntd",
    "bill_amount_2005_06_ntd", "bill_amount_2005_05_ntd", "bill_amount_2005_04_ntd",
    "paid_amount_2005_09_ntd", "paid_amount_2005_08_ntd", "paid_amount_2005_07_ntd",
    "paid_amount_2005_06_ntd", "paid_amount_2005_05_ntd", "paid_amount_2005_04_ntd",
]
NEW_FEATURES = [
    "utilization_09", "avg_utilization_6m", "repayment_ratio_6m",
    "trend_pay_status", "overdue_count_6m", "bill_std_6m",
]
FEATURES = BASE_FEATURES + NEW_FEATURES
LABEL = "default_next_month"
MIN_RECALL = 0.70


def pick_threshold(y_true, proba, min_recall=MIN_RECALL):
    """验证集选阈值：召回率 ≥ min_recall 的候选中精确率最高者。"""
    order = np.argsort(-proba)
    y_sorted = np.asarray(y_true)[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / max(y_sorted.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
    ok = np.where(recall >= min_recall)[0]
    if len(ok) == 0:
        return float(proba[order][-1])
    best = ok[np.argmax(precision[ok])]
    return float(proba[order][best])


def evaluate(y_true, proba, threshold):
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


def main():
    t0 = time.time()
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "valid"]
    te = df[df["split"] == "test"]
    Xtr, ytr = tr[FEATURES], tr[LABEL]
    Xva, yva = va[FEATURES], va[LABEL]
    Xte, yte = te[FEATURES], te[LABEL]
    print(f"数据：{len(df)} 行；训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}")

    # ---------------- 1. 逻辑回归（第二个模型，满足“至少两个模型”）
    print("训练逻辑回归 …")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(Xtr, ytr)
    pv_lr = lr.predict_proba(Xva)[:, 1]
    pt_lr = lr.predict_proba(Xte)[:, 1]
    joblib.dump(lr, OUT / "lr.pkl")

    # ---------------- 2. 调参前的随机森林（默认参数，同一份特征）
    print("训练随机森林（调参前）…")
    rf_base = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_base.fit(Xtr, ytr)
    pv_rf_base = rf_base.predict_proba(Xva)[:, 1]
    # rf_base 只用于「调参前后」对比，界面不需要该模型文件，故不落盘（避免 130MB+ 产物）

    # ---------------- 3. RandomizedSearchCV 调参（重点拉召回）
    print("随机搜索调参（n_iter=10, cv=3）…")
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 8, 10, 14, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist, n_iter=10, cv=3,
        scoring="roc_auc", random_state=42, n_jobs=1, verbose=0,
    )
    search.fit(Xtr, ytr)
    rf_tuned = search.best_estimator_
    pv_rf = rf_tuned.predict_proba(Xva)[:, 1]
    pt_rf = rf_tuned.predict_proba(Xte)[:, 1]
    joblib.dump(rf_tuned, OUT / "rf_tuned.pkl")

    # 调参日志：每一组参数 + 交叉验证得分
    trials = []
    for i, params in enumerate(search.cv_results_["params"]):
        trials.append({
            "rank": int(search.cv_results_["rank_test_score"][i]),
            "mean_cv_auc": float(search.cv_results_["mean_test_score"][i]),
            "std_cv_auc": float(search.cv_results_["std_test_score"][i]),
            "params": {k: (None if v is None else (v if not isinstance(v, (np.floating, np.integer)) else v.item()))
                       for k, v in params.items()},
        })
    trials.sort(key=lambda x: x["rank"])
    tuning_log = {
        "search_method": "RandomizedSearchCV",
        "n_iter": 10, "cv": 3, "scoring": "roc_auc", "random_state": 42,
        "search_space": {k: [str(v) for v in vals] for k, vals in param_dist.items()},
        "best_params": {k: (None if v is None else v) for k, v in search.best_params_.items()},
        "best_cv_auc": float(search.best_score_),
        "trials": trials,
    }
    (OUT / "tuning_log.json").write_text(
        json.dumps(tuning_log, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 4. 进阶树模型：梯度提升树（HistGradientBoosting）
    print("训练梯度提升树 …")
    gbdt = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=42)
    gbdt.fit(Xtr, ytr)
    pv_gb = gbdt.predict_proba(Xva)[:, 1]
    pt_gb = gbdt.predict_proba(Xte)[:, 1]
    joblib.dump(gbdt, OUT / "gbdt.pkl")

    # ---------------- 5. 阈值与指标
    models_valid = {"逻辑回归": pv_lr, "随机森林（调参前）": pv_rf_base,
                    "随机森林（调参后）": pv_rf, "梯度提升树": pv_gb}
    models_test = {"逻辑回归": pt_lr, "随机森林（调参后）": pt_rf, "梯度提升树": pt_gb}

    thresholds = {name: pick_threshold(yva, p) for name, p in models_valid.items()}
    metrics = {"valid": {}, "test": {}}
    for name, p in models_valid.items():
        metrics["valid"][name] = {
            "at_0.5": evaluate(yva, p, 0.5),
            "at_threshold": evaluate(yva, p, thresholds[name]),
        }
    for name, p in models_test.items():
        metrics["test"][name] = {"at_threshold": evaluate(yte, p, thresholds[name])}
    metrics["settings"] = {
        "data": CSV.name, "features": FEATURES, "n_base": len(BASE_FEATURES),
        "n_new": len(NEW_FEATURES), "min_recall": MIN_RECALL,
        "split": {"train": len(tr), "valid": len(va), "test": len(te)},
        "best_params": tuning_log["best_params"], "best_cv_auc": tuning_log["best_cv_auc"],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "metrics_v2.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "feature_list.json").write_text(
        json.dumps({"base": BASE_FEATURES, "new": NEW_FEATURES}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # 调参前后对比（验证集，阈值 0.5）
    rows = []
    for label, name in [("调参前（默认参数）", "随机森林（调参前）"),
                        ("调参后（RandomizedSearchCV）", "随机森林（调参后）")]:
        m = metrics["valid"][name]["at_0.5"]
        rows.append({"阶段": label, "Accuracy": round(m["accuracy"], 4),
                     "Precision": round(m["precision"], 4), "Recall": round(m["recall"], 4),
                     "F1": round(m["f1"], 4), "AUC": round(m["auc"], 4)})
    pd.DataFrame(rows).to_csv(OUT / "tuning_before_after.csv", index=False, encoding="utf-8-sig")

    # 三模型横向对比（验证集阈值 0.5）
    rows = []
    for name in ["逻辑回归", "随机森林（调参后）", "梯度提升树"]:
        m = metrics["valid"][name]["at_0.5"]
        rows.append({"模型": name, "Accuracy": round(m["accuracy"], 4),
                     "Precision": round(m["precision"], 4), "Recall": round(m["recall"], 4),
                     "F1": round(m["f1"], 4), "AUC": round(m["auc"], 4)})
    pd.DataFrame(rows).to_csv(OUT / "model_comparison_v2.csv", index=False, encoding="utf-8-sig")

    # 验证集概率（供 C 岗统一阈值口径）
    pd.DataFrame({
        "record_id": va["record_id"].values, "y_true": yva.values,
        "proba_rf": pv_rf, "proba_lr": pv_lr, "proba_gbdt": pv_gb,
    }).to_csv(OUT / "valid_proba.csv", index=False, encoding="utf-8-sig")

    print("\n===== 调参前后（验证集，阈值 0.5）=====")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n===== 三模型（验证集，阈值 0.5）=====")
    for name in ["逻辑回归", "随机森林（调参前）", "随机森林（调参后）", "梯度提升树"]:
        m = metrics["valid"][name]["at_0.5"]
        print(f"{name}: AUC {m['auc']:.4f}  Recall {m['recall']:.4f}  "
              f"Prec {m['precision']:.4f}  F1 {m['f1']:.4f}")
    print("\n===== 阈值（验证集，召回≥70% 中精确率最高）=====")
    for name, th in thresholds.items():
        print(f"{name}: {th:.4f}")
    print("\n===== 测试集（仅评价一次）=====")
    for name in ["逻辑回归", "随机森林（调参后）", "梯度提升树"]:
        m = metrics["test"][name]["at_threshold"]
        print(f"{name}: AUC {m['auc']:.4f}  Recall {m['recall']:.4f}  Prec {m['precision']:.4f}  "
              f"F1 {m['f1']:.4f}  TN{m['tn']} FP{m['fp']} FN{m['fn']} TP{m['tp']}")
    print(f"\n最优参数: {tuning_log['best_params']}  CV-AUC {tuning_log['best_cv_auc']:.4f}")
    print(f"耗时 {time.time() - t0:.1f}s；产物目录 {OUT}")


if __name__ == "__main__":
    main()
