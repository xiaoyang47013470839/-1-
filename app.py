# -*- coding: utf-8 -*-
"""
信用卡客户下月违约风险分析系统 v2（计算机金融学导论 · 课程项目一 · 界面模块）

运行：
    streamlit run app.py

六个页面（st.navigation 多页面模式）：
    1. 📊 数据概览      统计卡片 + 三集划分 + 三类分布图 + 6 个衍生特征
    2. ⚖️ 模型比较      三个模型（LR / 调参后 RF / 梯度提升树）一键切换 + ROC 叠加
    3. 🔧 调参过程      搜索空间、每一组参数得分、最优参数、调参前后对比
    4. ✅ 最终评价      锁定模型 + 验证集阈值，对测试集只评价一次
    5. 👤 客户风险演示  单客户概率 + 阈值判定 + 局部特征贡献（SHAP 置换近似）
    6. 📦 批量预测      按数据范围批量打分并导出 CSV

数据与模型：
    data/03_credit_default_features.csv（A 岗交付：清洗 + 6 个业务衍生特征）
    models_v2/*.pkl、metrics_v2.json、tuning_log.json（本目录 train_models_v2.py 产出）
    口径：预处理/模型只在训练集拟合，阈值只用验证集选择，测试集只评价一次。
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "03_credit_default_features.csv"
MODEL_DIR = BASE_DIR / "models_v2"
METRICS_JSON = MODEL_DIR / "metrics_v2.json"
TUNING_JSON = MODEL_DIR / "tuning_log.json"
FEATURE_JSON = MODEL_DIR / "feature_list.json"

PAY_MONTHS = ["2005_09", "2005_08", "2005_07", "2005_06", "2005_05", "2005_04"]
PAY_MONTH_CN = {"2005_09": "2005年9月", "2005_08": "2005年8月", "2005_07": "2005年7月",
                "2005_06": "2005年6月", "2005_05": "2005年5月", "2005_04": "2005年4月"}

SEX_CN = {1: "男", 2: "女"}
EDU_CN = {1: "研究生及以上", 2: "本科", 3: "高中", 4: "其他"}
MAR_CN = {1: "已婚", 2: "未婚", 3: "其他"}

FEATURE_CN = {
    "credit_limit_ntd": "授信额度（新台币）", "sex_code": "性别", "education_code": "学历",
    "marriage_code": "婚姻状况", "age_years": "年龄（岁）",
    "pay_status_2005_09": "9月还款状态", "pay_status_2005_08": "8月还款状态",
    "pay_status_2005_07": "7月还款状态", "pay_status_2005_06": "6月还款状态",
    "pay_status_2005_05": "5月还款状态", "pay_status_2005_04": "4月还款状态",
    "bill_amount_2005_09_ntd": "9月账单金额", "bill_amount_2005_08_ntd": "8月账单金额",
    "bill_amount_2005_07_ntd": "7月账单金额", "bill_amount_2005_06_ntd": "6月账单金额",
    "bill_amount_2005_05_ntd": "5月账单金额", "bill_amount_2005_04_ntd": "4月账单金额",
    "paid_amount_2005_09_ntd": "9月还款金额", "paid_amount_2005_08_ntd": "8月还款金额",
    "paid_amount_2005_07_ntd": "7月还款金额", "paid_amount_2005_06_ntd": "6月还款金额",
    "paid_amount_2005_05_ntd": "5月还款金额", "paid_amount_2005_04_ntd": "4月还款金额",
    "utilization_09": "最新一期额度使用率", "avg_utilization_6m": "近6期平均额度使用率",
    "repayment_ratio_6m": "近6期还款覆盖比", "trend_pay_status": "还款恶化趋势",
    "overdue_count_6m": "近6期逾期次数", "bill_std_6m": "近6期账单波动",
}

NEW_FEATURE_DESC = {
    "utilization_09": ("最新一期额度使用率", "9 月账单金额 ÷ 授信额度，衡量当期杠杆"),
    "avg_utilization_6m": ("近 6 期平均额度使用率", "六期使用率的均值，衡量长期资金紧张程度"),
    "repayment_ratio_6m": ("近 6 期还款覆盖比", "六期还款金额合计 ÷ 六期账单金额合计"),
    "trend_pay_status": ("还款恶化趋势", "最近一期还款状态减去最远一期，正值表示恶化"),
    "overdue_count_6m": ("近 6 期逾期次数", "六期还款状态中大于 0 的期数"),
    "bill_std_6m": ("近 6 期账单波动", "六期账单金额的标准差，衡量消费/负债波动"),
}

MODEL_ORDER = ["逻辑回归", "随机森林（调参后）", "梯度提升树"]
MODEL_COLOR = {"逻辑回归": "#4C78A8", "随机森林（调参后）": "#E45756", "梯度提升树": "#54A24B"}


def pay_status_cn(code: int) -> str:
    if code == -2:
        return "无消费"
    if code == -1:
        return "全额还款"
    if code == 0:
        return "循环信用"
    return f"逾期{int(code)}个月"


# ------------------------------------------------------------------ 缓存加载
@st.cache_data(show_spinner="正在读取 03_credit_default_features.csv（30,000 条 · 32 列）…")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_tuning() -> dict:
    return json.loads(TUNING_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_feature_list() -> dict:
    return json.loads(FEATURE_JSON.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="正在加载模型（逻辑回归 / 随机森林 / 梯度提升树）…")
def load_models() -> dict:
    """加载 models_v2 下训练好的模型。缺文件时抛出明确错误，由页面统一提示。"""
    needed = {"lr": "lr.pkl", "rf_tuned": "rf_tuned.pkl", "gbdt": "gbdt.pkl"}
    missing = [f for f in needed.values() if not (MODEL_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"models_v2 目录缺少模型文件：{'、'.join(missing)}。"
            "请先在项目目录运行 `python train_models_v2.py` 生成模型。")
    return {k: joblib.load(MODEL_DIR / v) for k, v in needed.items()}


def safe_page(fn):
    """统一兜底：数据/模型缺失时给出友好提示，而不是白屏报错。"""
    def wrapper():
        if not DATA_CSV.exists():
            st.error(f"找不到数据文件：{DATA_CSV.name}。请确认 data 目录中有该文件。")
            st.stop()
        try:
            fn()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:  # 不吞掉细节，方便现场排查
            st.error(f"页面加载失败：{type(e).__name__}: {e}")
            st.info("可在项目目录运行 `python train_models_v2.py` 重新生成 models_v2 产物后刷新。")
            st.stop()
    return wrapper


# ------------------------------------------------------------------ 通用绘图
def style_fig(fig, x_title, y_title, legend_title=None, height=420):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=50, r=24, t=50, b=45),
        legend_title_text=legend_title, font=dict(size=13),
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return fig


def confusion_fig(tn, fp, fn, tp):
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["预测：正常", "预测：违约"], y=["实际：正常", "实际：违约"],
        colorscale="Blues", showscale=False,
        text=[[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]],
        texttemplate="%{text}", textfont={"size": 20},
    ))
    fig.update_layout(template="plotly_white", height=360,
                      margin=dict(l=10, r=10, t=30, b=10))
    fig.update_yaxes(autorange="reversed")
    return fig


def roc_fig(y_true, probas: dict, height=430):
    fig = go.Figure()
    for name, p in probas.items():
        fpr, tpr, _ = _roc(y_true, p)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name,
                                 line=dict(width=2.6, color=MODEL_COLOR.get(name))))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="随机猜测（AUC 0.5）",
                             line=dict(width=1.5, dash="dash", color="#999999")))
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=50, r=20, t=40, b=45),
                      legend=dict(x=0.55, y=0.08, bgcolor="rgba(255,255,255,0.85)"),
                      font=dict(size=13))
    fig.update_xaxes(title_text="假阳率 FPR", range=[0, 1])
    fig.update_yaxes(title_text="真阳率 TPR（召回率）", range=[0, 1])
    return fig


def _roc(y_true, proba):
    from sklearn.metrics import roc_curve
    return roc_curve(y_true, proba)


def threshold_curve(y_true, proba):
    from sklearn.metrics import precision_score, recall_score
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    pre, rec = [], []
    for t in grid:
        pred = (proba >= t).astype(int)
        pre.append(precision_score(y_true, pred, zero_division=0))
        rec.append(recall_score(y_true, pred, zero_division=0))
    return grid, np.array(pre), np.array(rec)


# ------------------------------------------------------------------ 页面 1
@safe_page
def page_overview():
    df = load_data()
    feats = load_feature_list()
    st.title("📊 数据概览")
    st.caption("数据来源：03_credit_default_features.csv（A 岗交付：清洗后 30,000 条 + 6 个业务衍生特征）；"
               "训练 / 验证 / 测试 = 20,999 / 4,500 / 4,501")

    n_missing = int(df.isna().sum().sum())
    dup_ids = int(df["record_id"].duplicated().sum())
    rate = df["default_next_month"].mean()
    n_default = int(df["default_next_month"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("总样本数", f"{len(df):,}")
    c2.metric("总体违约率", f"{rate:.2%}")
    c3.metric("违约客户数", f"{n_default:,}")
    c4, c5 = st.columns(2)
    c4.metric("缺失值总数", n_missing)
    c5.metric("衍生特征数", len(feats["new"]))

    st.subheader("数据划分与各组违约率")
    split_stat = (df.groupby("split")
                  .agg(样本量=("record_id", "size"),
                       违约数=("default_next_month", "sum"),
                       违约率=("default_next_month", "mean"))
                  .reindex(["train", "valid", "test"])
                  .rename(index={"train": "训练集", "valid": "验证集", "test": "测试集"})
                  .reset_index())
    split_stat["违约率"] = split_stat["违约率"].map(lambda x: f"{x:.2%}")
    st.dataframe(split_stat, width="stretch", hide_index=True)

    plot_df = df.copy()
    plot_df["违约标签"] = plot_df["default_next_month"].map({0: "正常", 1: "违约"})
    cmap = {"正常": "#4C78A8", "违约": "#E45756"}

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(plot_df, x="credit_limit_ntd", color="违约标签", nbins=60,
                           barmode="overlay", opacity=0.6, color_discrete_map=cmap)
        style_fig(fig, "授信额度（新台币）", "客户数", "分组", height=370)
        st.plotly_chart(fig, width="stretch")
    with col_b:
        fig = px.histogram(plot_df, x="age_years", color="违约标签", nbins=45,
                           barmode="overlay", opacity=0.6, color_discrete_map=cmap)
        style_fig(fig, "年龄（岁）", "客户数", "分组", height=370)
        st.plotly_chart(fig, width="stretch")

    st.subheader("还款状态分布（可切换月份）")
    month = st.selectbox("选择月份", PAY_MONTHS, index=0,
                         format_func=lambda m: PAY_MONTH_CN[m])
    status_col = f"pay_status_{month}"
    stat = df.groupby(status_col).size().reset_index(name="客户数").sort_values(status_col)
    stat["状态说明"] = stat[status_col].map(pay_status_cn)
    fig = px.bar(stat, x=status_col, y="客户数", color="客户数",
                 color_continuous_scale="Blues", text_auto=True)
    style_fig(fig, "还款状态编码", "客户数", height=370)
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("新增衍生特征分布")
    pick = st.selectbox("选择衍生特征", feats["new"],
                        format_func=lambda f: f"{FEATURE_CN.get(f, f)}（{f}）")
    fig = px.histogram(plot_df, x=pick, color="违约标签", nbins=50,
                       barmode="overlay", opacity=0.6, color_discrete_map=cmap)
    style_fig(fig, FEATURE_CN.get(pick, pick), "客户数", "分组", height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption("对比可见：违约客户的额度使用率、逾期次数普遍更高，还款覆盖比更低——"
               "这些业务特征把原始账单/还款数字里“看不出来”的风险信号显性化了。")

    with st.expander("📖 字段编码与衍生特征说明（点击展开）"):
        st.markdown("**还款状态编码**：`-2` 无消费、`-1` 全额还款、`0` 循环信用、"
                    "`1..8` 逾期月数。负值是业务编码，不是缺失值。")
        rows = [{"特征名": k, "中文名": v[0], "计算口径": v[1]} for k, v in NEW_FEATURE_DESC.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ------------------------------------------------------------------ 页面 2
@safe_page
def page_compare():
    df = load_data()
    feats = load_feature_list()
    metrics = load_metrics()
    va = df[df["split"] == "valid"]
    y = va["default_next_month"].to_numpy()

    st.title("⚖️ 模型比较（验证集）")
    st.caption("三个模型都在训练集 20,999 条上拟合，在验证集 4,500 条上比较；"
               "测试集不参与本页任何计算。")

    st.subheader("ROC 曲线叠加")
    probas = _valid_probas(va)
    st.plotly_chart(roc_fig(y, probas), width="stretch")
    st.caption("三条曲线同图对比：随机森林（调参后）与梯度提升树明显高于逻辑回归，"
               "说明树模型能捕捉特征间的非线性交互。")

    st.subheader("指标对比（统一阈值 0.5）")
    rows = []
    for name in MODEL_ORDER:
        m = metrics["valid"][name]["at_0.5"]
        rows.append({"模型": name, "AUC": round(m["auc"], 4), "准确率": round(m["accuracy"], 4),
                     "精确率": round(m["precision"], 4), "召回率": round(m["recall"], 4),
                     "F1": round(m["f1"], 4)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("单模型细看（可切换）")
    model = st.radio("选择模型", MODEL_ORDER, horizontal=True, index=1,
                     format_func=lambda x: f"{x}")
    m05 = metrics["valid"][model]["at_0.5"]
    mt = metrics["valid"][model]["at_threshold"]

    c1, c2, c3 = st.columns(3)
    c1.metric("验证集 AUC", f"{m05['auc']:.4f}")
    c2.metric("精确率 @0.5", f"{m05['precision']:.4f}")
    c3.metric("召回率 @0.5", f"{m05['recall']:.4f}")
    c4, c5 = st.columns(2)
    c4.metric("验证集自动阈值", f"{mt['threshold']:.4f}")
    c5.metric(f"{model} F1 @0.5", f"{m05['f1']:.4f}")

    st.markdown(f"**阈值滑块**：拖动查看「{model}」在验证集上的权衡；"
                f"系统自动选出的报警阈值是 **{mt['threshold']:.4f}**"
                f"（规则：召回率 ≥ 70% 的候选中精确率最高）。")
    p = probas[model]
    t = st.slider("报警阈值", 0.01, 0.99, float(round(mt["threshold"], 2)), 0.01)
    pred = (p >= t).astype(int)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    pre = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)

    b1, b2, b3 = st.columns(3)
    b1.metric("精确率", f"{pre:.2%}")
    b2.metric("召回率", f"{rec:.2%}")
    b3.metric("报警人数", f"{int(pred.sum()):,}")
    b4, b5 = st.columns(2)
    b4.metric("正确拦截", f"{tp:,}")
    b5.metric("漏判", f"{fn:,}")

    grid, ps, rs = _threshold_cache(tuple(np.round(p, 6)), tuple(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=ps, name="精确率", line=dict(color="#E45756", width=2.4)))
    fig.add_trace(go.Scatter(x=grid, y=rs, name="召回率", line=dict(color="#4C78A8", width=2.4)))
    fig.add_vline(x=t, line_dash="dash", line_color="#333333",
                  annotation_text=f"当前 {t:.2f}", annotation_position="top")
    style_fig(fig, "报警阈值", "指标值", "曲线", height=400)
    st.plotly_chart(fig, width="stretch")
    st.caption("阈值升高 → 报警更少、更准，但漏判更多；阈值降低 → 抓得更全，但误报上升。")


@st.cache_data(show_spinner=False)
def _threshold_cache(proba_tuple, y_tuple):
    return threshold_curve(np.array(y_tuple), np.array(proba_tuple))


@st.cache_data(show_spinner="正在计算验证集预测概率…")
def _valid_probas(va: pd.DataFrame) -> dict:
    models = load_models()
    feats = load_feature_list()
    X = va[feats["base"] + feats["new"]]
    return {
        "逻辑回归": models["lr"].predict_proba(X)[:, 1],
        "随机森林（调参后）": models["rf_tuned"].predict_proba(X)[:, 1],
        "梯度提升树": models["gbdt"].predict_proba(X)[:, 1],
    }


# ------------------------------------------------------------------ 页面 3
@safe_page
def page_tuning():
    log = load_tuning()
    metrics = load_metrics()
    st.title("🔧 调参过程")
    st.caption("目标：把违约类召回率拉上来（初版 baseline 的召回只有 0.36，漏判严重）。")

    c1, c2, c3 = st.columns(3)
    c1.metric("搜索方法", log["search_method"])
    c2.metric("搜索组合数", f"{log['n_iter']} × {log['cv']} 折")
    c3.metric("最优交叉验证 AUC", f"{log['best_cv_auc']:.4f}")

    st.subheader("搜索空间")
    space = [{"参数": k, "候选值": "、".join(v)} for k, v in log["search_space"].items()]
    st.dataframe(pd.DataFrame(space), width="stretch", hide_index=True)

    st.subheader("最优参数组合")
    best = log["best_params"]
    st.dataframe(pd.DataFrame([{"参数": k, "取值": str(v)} for k, v in best.items()]),
                 width="stretch", hide_index=True)

    st.subheader("每一组参数的交叉验证得分")
    rows = [{"排名": t["rank"], "CV-AUC 均值": round(t["mean_cv_auc"], 4),
             "标准差": round(t["std_cv_auc"], 4),
             **{k: str(v) for k, v in t["params"].items()}} for t in log["trials"]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("调参前后对比（验证集，阈值 0.5）")
    before = metrics["valid"]["随机森林（调参前）"]["at_0.5"]
    after = metrics["valid"]["随机森林（调参后）"]["at_0.5"]
    rows = []
    for label, m in [("调参前（默认参数）", before), ("调参后（RandomizedSearchCV）", after)]:
        rows.append({"阶段": label, "AUC": round(m["auc"], 4), "准确率": round(m["accuracy"], 4),
                     "精确率": round(m["precision"], 4), "召回率": round(m["recall"], 4),
                     "F1": round(m["f1"], 4)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    comp = pd.DataFrame({
        "指标": ["AUC", "召回率", "精确率", "F1"],
        "调参前": [before["auc"], before["recall"], before["precision"], before["f1"]],
        "调参后": [after["auc"], after["recall"], after["precision"], after["f1"]],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp["指标"], y=comp["调参前"], name="调参前", marker_color="#A0AAB5"))
    fig.add_trace(go.Bar(x=comp["指标"], y=comp["调参后"], name="调参后", marker_color="#E45756"))
    fig.update_layout(barmode="group", template="plotly_white", height=380,
                      margin=dict(l=50, r=20, t=40, b=45), font=dict(size=13))
    fig.update_yaxes(title_text="指标值", range=[0, 0.9])
    st.plotly_chart(fig, width="stretch")

    st.success(f"召回率从 {before['recall']:.2%} 提升到 {after['recall']:.2%}"
               f"（+{after['recall'] - before['recall']:.2%}），"
               f"F1 从 {before['f1']:.4f} 提升到 {after['f1']:.4f}；"
               f"代价是精确率从 {before['precision']:.2%} 降到 {after['precision']:.2%}。"
               "这正是“少漏判”目标下的主动取舍。")


# ------------------------------------------------------------------ 页面 4
@safe_page
def page_final():
    df = load_data()
    metrics = load_metrics()
    st.title("✅ 最终评价（测试集，仅评价一次）")
    st.caption("锁定模型：随机森林（调参后）；阈值在验证集上选定后固定，测试集只用来报告结果。")

    mt = metrics["test"]["随机森林（调参后）"]["at_threshold"]
    c1, c2, c3 = st.columns(3)
    c1.metric("测试集 ROC-AUC", f"{mt['auc']:.4f}")
    c2.metric("违约类精确率", f"{mt['precision']:.4f}")
    c3.metric("违约类召回率", f"{mt['recall']:.4f}")
    c4, c5 = st.columns(2)
    c4.metric("报警阈值（验证集选定）", f"{mt['threshold']:.4f}")
    c5.metric("测试集样本", f"{mt['n']:,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**混淆矩阵**")
        st.plotly_chart(confusion_fig(mt["tn"], mt["fp"], mt["fn"], mt["tp"]), width="stretch")
    with col_b:
        st.markdown("**ROC 曲线**")
        te = df[df["split"] == "test"]
        feats = load_feature_list()
        models = load_models()
        X = te[feats["base"] + feats["new"]]
        p = models["rf_tuned"].predict_proba(X)[:, 1]
        st.plotly_chart(roc_fig(te["default_next_month"].to_numpy(),
                                {"随机森林（调参后）": p}), width="stretch")

    d1, d2, d3 = st.columns(3)
    d1.metric("正确拦截（TP）", f"{mt['tp']:,}")
    d2.metric("误判（FP）", f"{mt['fp']:,}")
    d3.metric("漏判（FN）", f"{mt['fn']:,}")

    st.subheader("三个模型在测试集上的同一口径对比")
    rows = []
    for name in MODEL_ORDER:
        m = metrics["test"][name]["at_threshold"]
        rows.append({"模型": name, "AUC": round(m["auc"], 4), "精确率": round(m["precision"], 4),
                     "召回率": round(m["recall"], 4), "F1": round(m["f1"], 4),
                     "阈值": round(m["threshold"], 4)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("三个模型都用各自验证集选出的阈值在测试集上评价一次；"
               "调参后随机森林的测试集 AUC 由初版的 0.7587 提升到 "
               f"{metrics['test']['随机森林（调参后）']['at_threshold']['auc']:.4f}。")


# ------------------------------------------------------------------ 页面 5
@safe_page
def page_demo():
    df = load_data()
    feats = load_feature_list()
    metrics = load_metrics()
    models = load_models()

    st.title("👤 客户风险演示")
    st.caption("输入客户编号或随机抽取一条记录，查看违约概率、阈值判定与主要风险因子。")

    col1, col2 = st.columns([2, 1])
    with col1:
        rid = st.number_input("客户编号（1 – 30000）", min_value=1, max_value=int(df["record_id"].max()),
                              value=6, step=1)
    with col2:
        st.write("")
        st.write("")
        if st.button("🎲 随机抽取一条", width="stretch"):
            st.session_state["demo_rid"] = int(np.random.choice(df["record_id"].values))
    if "demo_rid" in st.session_state:
        rid = int(st.session_state["demo_rid"])

    sub = df[df["record_id"] == rid]
    if sub.empty:
        st.warning(f"没有编号为 {rid} 的客户。有效编号范围：1 – {int(df['record_id'].max()):,}，"
                   "可输入其他编号或点“随机抽取一条”。")
        st.stop()

    rec = sub.iloc[0]
    X = sub[feats["base"] + feats["new"]]
    proba = float(models["rf_tuned"].predict_proba(X)[0, 1])
    threshold = float(metrics["test"]["随机森林（调参后）"]["at_threshold"]["threshold"])

    b1, b2, b3 = st.columns(3)
    b1.metric("授信额度", f"{rec['credit_limit_ntd'] / 10000:.1f} 万新台币")
    b2.metric("年龄", f"{rec['age_years']} 岁")
    b3.metric("性别", SEX_CN.get(int(rec["sex_code"]), "未知"))
    b4, b5 = st.columns(2)
    b4.metric("学历", EDU_CN.get(int(rec["education_code"]), "其他"))
    b5.metric("婚姻", MAR_CN.get(int(rec["marriage_code"]), "其他"))

    r1, r2, r3 = st.columns(3)
    r1.metric("违约概率", f"{proba:.2%}")
    r2.metric("报警阈值", f"{threshold:.2%}")
    r3.metric("额度使用率（最新）", f"{rec['utilization_09']:.2f}")
    if proba >= threshold:
        st.error(f"⚠️ 高风险：违约概率 {proba:.2%} ≥ 报警阈值 {threshold:.2%}，建议进入人工复核。")
    else:
        st.success(f"✅ 暂未触发报警：违约概率 {proba:.2%} < 报警阈值 {threshold:.2%}。")

    st.subheader("为什么给出这个判断（局部特征贡献）")
    st.caption("以训练集各特征的中位数为基准，用置换采样估计每个特征对该客户概率的贡献"
               "（模型无关的 SHAP 近似）。条形向右表示推高风险，向左表示降低风险。")
    base_p = _baseline_proba()
    contrib = _local_contrib(rid)
    st.markdown(f"基准（训练集中位数客户）违约概率 **{base_p:.2%}** → 该客户 **{proba:.2%}**，"
                f"差值由下列 {len(contrib)} 个特征的贡献累加得到。")
    top = contrib.head(9).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["贡献"], y=top["特征"], orientation="h",
        marker_color=["#E45756" if v > 0 else "#4C78A8" for v in top["贡献"]],
        text=[f"{v:+.2%}" for v in top["贡献"]], textposition="outside",
    ))
    fig.update_layout(template="plotly_white", height=420,
                      margin=dict(l=130, r=60, t=30, b=45), font=dict(size=13))
    fig.update_xaxes(title_text="对违约概率的贡献（百分点）")
    fig.update_yaxes(title_text="")
    st.plotly_chart(fig, width="stretch")

    if rec["split"] == "test":
        truth = int(rec["default_next_month"])
        st.info(f"该记录属于**测试集**。事后真实标签：{'违约' if truth else '未违约'}；"
                f"模型判定：{'报警' if proba >= threshold else '未报警'}。"
                "两者分栏并列展示，避免把事后结果误读成模型能力。")
    else:
        st.caption(f"该记录属于「{ {'train': '训练集', 'valid': '验证集'}.get(rec['split'], rec['split']) }」，"
                   "其真实标签不单独展示，以免与模型预测混淆。")


@st.cache_data(show_spinner=False)
def _baseline_proba() -> float:
    """训练集中位数客户（基准样本）的违约概率，用于解释局部贡献的起点。"""
    df = load_data()
    feats = load_feature_list()
    models = load_models()
    cols = feats["base"] + feats["new"]
    base = df[df["split"] == "train"][cols].median().to_frame().T
    return float(models["rf_tuned"].predict_proba(base)[0, 1])


@st.cache_data(show_spinner="正在计算该客户的特征贡献（置换采样 SHAP）…")
def _local_contrib(record_id: int) -> pd.DataFrame:
    df = load_data()
    feats = load_feature_list()
    models = load_models()
    cols = feats["base"] + feats["new"]
    tr = df[df["split"] == "train"]
    base = tr[cols].median().to_numpy(dtype=float)
    x = df[df["record_id"] == record_id][cols].iloc[0].to_numpy(dtype=float)

    rng = np.random.default_rng(42)
    m = 40
    n_f = len(cols)
    rows, owner = [], []
    for _ in range(m):
        perm = rng.permutation(n_f)
        cur = base.copy()
        rows.append(cur.copy())
        owner.append(-1)
        for f in perm:
            cur = cur.copy()
            cur[f] = x[f]
            rows.append(cur)
            owner.append(f)
    proba = models["rf_tuned"].predict_proba(np.vstack(rows))[:, 1].reshape(m, n_f + 1)
    phi = np.zeros(n_f)
    for i in range(m):
        phi += np.diff(proba[i])
    phi /= m
    out = pd.DataFrame({"特征": [FEATURE_CN.get(c, c) for c in cols],
                        "贡献": phi, "列名": cols})
    out["绝对值"] = out["贡献"].abs()
    return out.sort_values("绝对值", ascending=False).drop(columns="绝对值").reset_index(drop=True)


# ------------------------------------------------------------------ 页面 6
@safe_page
def page_batch():
    df = load_data()
    feats = load_feature_list()
    metrics = load_metrics()
    models = load_models()
    cols = feats["base"] + feats["new"]

    st.title("📦 批量预测与导出")
    st.caption("按数据范围批量打分，可导出 CSV 交给组员做进一步分析或留档。"
               "首次进入会自动生成验证集预览；切换范围或模型后点上方按钮重新计算。")

    c1, c2 = st.columns([2, 1])
    with c1:
        scope = st.selectbox("选择预测范围", ["验证集（4,500 条）", "测试集（4,501 条）", "全量（30,000 条）"])
    with c2:
        model_name = st.selectbox("选择模型", MODEL_ORDER, index=1)

    key = {"验证集（4,500 条）": "valid", "测试集（4,501 条）": "test", "全量（30,000 条）": None}[scope]
    sub = df if key is None else df[df["split"] == key]
    thr = float(metrics["valid"][model_name]["at_threshold"]["threshold"])

    clicked = st.button("🚀 开始批量预测", type="primary")
    first_visit = "batch_meta" not in st.session_state
    if clicked or first_visit:
        if len(sub) > 6000:
            bar = st.progress(0.0, text="正在分批打分…")
            parts = []
            step = 6000
            for i in range(0, len(sub), step):
                chunk = sub.iloc[i:i + step]
                p = _predict(models, model_name, chunk[cols])
                parts.append(pd.DataFrame({"record_id": chunk["record_id"].values,
                                           "违约概率": p}))
                bar.progress((i + len(chunk)) / len(sub), text=f"已打分 {i + len(chunk):,} / {len(sub):,}")
            pred_df = pd.concat(parts, ignore_index=True)
            bar.empty()
        else:
            with st.spinner(f"正在对 {len(sub):,} 条记录打分…"):
                pred_df = pd.DataFrame({
                    "record_id": sub["record_id"].values,
                    "违约概率": _predict(models, model_name, sub[cols]),
                })
        pred_df["是否报警"] = np.where(pred_df["违约概率"] >= thr, "报警", "未报警")
        st.session_state["batch_df"] = pred_df
        st.session_state["batch_meta"] = (scope, model_name, thr)

    if "batch_df" in st.session_state:
        pred_df = st.session_state["batch_df"]
        scope_used, model_used, thr_used = st.session_state["batch_meta"]
        n_alarm = int((pred_df["是否报警"] == "报警").sum())
        a1, a2, a3 = st.columns(3)
        a1.metric("记录数", f"{len(pred_df):,}")
        a2.metric("报警人数", f"{n_alarm:,}")
        a3.metric("报警占比", f"{n_alarm / max(len(pred_df), 1):.2%}")
        st.caption(f"口径：{model_used}；阈值 {thr_used:.4f}（来自验证集）；范围 {scope_used}。")
        st.dataframe(pred_df.head(200), width="stretch", hide_index=True)
        st.download_button(
            "⬇️ 导出预测结果 CSV",
            data=pred_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"预测结果_{model_used}_{scope_used}.csv",
            mime="text/csv",
        )

    with st.expander("📤 也可以上传自己的 CSV 批量预测"):
        st.caption(f"需包含 {len(cols)} 个特征列与 record_id；不需要标签列。")
        up = st.file_uploader("选择 CSV 文件", type=["csv"])
        if up is not None:
            try:
                new_df = pd.read_csv(up, encoding="utf-8-sig")
                miss = [c for c in cols if c not in new_df.columns]
                if miss:
                    st.warning(f"缺少 {len(miss)} 个特征列，例如：{'、'.join(miss[:5])}。"
                               "请使用导出模板或与本项目数据同结构的文件。")
                else:
                    p = _predict(models, model_name, new_df[cols])
                    out = pd.DataFrame({
                        "record_id": new_df["record_id"].values if "record_id" in new_df else np.arange(len(new_df)),
                        "违约概率": p,
                    })
                    out["是否报警"] = np.where(out["违约概率"] >= thr, "报警", "未报警")
                    st.success(f"已对 {len(out):,} 条记录完成预测。")
                    st.dataframe(out.head(100), width="stretch", hide_index=True)
                    st.download_button("⬇️ 导出上传文件的预测结果", data=out.to_csv(index=False).encode("utf-8-sig"),
                                       file_name="上传文件_预测结果.csv", mime="text/csv")
            except Exception as e:
                st.error(f"文件解析失败：{type(e).__name__}: {e}。请确认是 UTF-8 编码的 CSV。")


def _predict(models: dict, name: str, X: pd.DataFrame) -> np.ndarray:
    key = {"逻辑回归": "lr", "随机森林（调参后）": "rf_tuned", "梯度提升树": "gbdt"}[name]
    return models[key].predict_proba(X)[:, 1]


# ------------------------------------------------------------------ 入口
def main():
    st.set_page_config(page_title="信用卡违约风险分析系统", page_icon="💳", layout="wide")
    st.sidebar.title("💳 信用卡违约风险")
    st.sidebar.caption("计算机金融学导论 · 课程项目一")
    st.sidebar.divider()
    st.sidebar.markdown(
        "**口径约定**\n\n"
        "- 预处理与模型只在训练集拟合\n"
        "- 阈值只用验证集选择\n"
        "- 测试集只评价一次\n\n"
        "**合规提示**\n\n"
        "相关性不等于因果；结果仅作教学示例，不用于真实授信决策。"
    )

    pages = [
        st.Page(page_overview, title="数据概览", icon="📊", url_path="overview", default=True),
        st.Page(page_compare, title="模型比较", icon="⚖️", url_path="compare"),
        st.Page(page_tuning, title="调参过程", icon="🔧", url_path="tuning"),
        st.Page(page_final, title="最终评价", icon="✅", url_path="final"),
        st.Page(page_demo, title="客户风险演示", icon="👤", url_path="demo"),
        st.Page(page_batch, title="批量预测", icon="📦", url_path="batch"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
