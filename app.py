# -*- coding: utf-8 -*-
"""
信用卡客户下月违约风险分析系统（计算机金融学导论 · 课程项目一 · 界面模块）

运行方式：
    streamlit run app.py

页面结构（左侧边栏切换，st.navigation 多页面模式）：
    1. 数据概览      —— 统计卡片 + 分布图表（授信额度 / 年龄 / 还款状态）
    2. 模型比较      —— 基线 / 逻辑回归 / 随机森林 在验证集上的对比与阈值探索
    3. 最终评价      —— 锁定模型 + 阈值 后对测试集的一次性评价结果
    4. 客户风险演示  —— 选择一条客户记录，输出风险概率、阈值与风险提示

分工说明：
    界面部分由本文件实现；模型文件 rf_best_model.pkl 与
    test_metrics.json / test_predictions.csv 由负责「模型训练」「测试优化」
    的组员提供，本界面只做加载与展示，不修改模型口径。
"""

import bz2
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 路径与常量
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "01_credit_default.csv"
MODEL_PKL = BASE_DIR / "models" / "rf_best_model.pkl"
MODEL_BZ2 = BASE_DIR / "models" / "rf_best_model.pkl.bz2"
METRICS_JSON = BASE_DIR / "models" / "test_metrics.json"
PRED_CSV = BASE_DIR / "models" / "test_predictions.csv"

# 模型使用的 23 个特征列（与 pkl 中 feature_names_in_ 一致）
FEATURES = [
    "credit_limit_ntd", "sex_code", "education_code", "marriage_code", "age_years",
    "pay_status_2005_09", "pay_status_2005_08", "pay_status_2005_07",
    "pay_status_2005_06", "pay_status_2005_05", "pay_status_2005_04",
    "bill_amount_2005_09_ntd", "bill_amount_2005_08_ntd", "bill_amount_2005_07_ntd",
    "bill_amount_2005_06_ntd", "bill_amount_2005_05_ntd", "bill_amount_2005_04_ntd",
    "paid_amount_2005_09_ntd", "paid_amount_2005_08_ntd", "paid_amount_2005_07_ntd",
    "paid_amount_2005_06_ntd", "paid_amount_2005_05_ntd", "paid_amount_2005_04_ntd",
]

# 还款状态月份（从近到远）
PAY_MONTHS = ["2005_09", "2005_08", "2005_07", "2005_06", "2005_05", "2005_04"]
PAY_MONTH_CN = {"2005_09": "2005年9月", "2005_08": "2005年8月", "2005_07": "2005年7月",
                "2005_06": "2005年6月", "2005_05": "2005年5月", "2005_04": "2005年4月"}

# 分类编码 → 中文（用于展示；编码含义参考课程数据字典）
SEX_CN = {1: "男", 2: "女"}
EDU_CN = {1: "研究生及以上", 2: "本科", 3: "高中", 4: "其他", 0: "未知", 5: "未知", 6: "未知"}
MAR_CN = {1: "已婚", 2: "未婚", 3: "其他", 0: "其他"}

PAY_STATUS_MIN_RECALL = 0.70  # 阈值选择约束：验证集召回率 ≥ 70%


def pay_status_cn(code: int) -> str:
    """还款状态编码 → 中文说明。-2/-1/0 为正常编码，不是缺失值。"""
    if code == -2:
        return "无消费"
    if code == -1:
        return "全额还款"
    if code == 0:
        return "循环信用"
    return f"逾期{code}个月"


# ---------------------------------------------------------------- 缓存加载
@st.cache_data(show_spinner="正在读取课程数据（约 3 万条）…")
def load_data() -> pd.DataFrame:
    """读取信用违约数据集（UTF-8 BOM，Excel 可直接打开的格式）。"""
    return pd.read_csv(DATA_CSV, encoding="utf-8-sig")


@st.cache_resource(show_spinner="正在加载组员交付的锁定模型…")
def load_locked_model():
    """加载第三棒组员交付的随机森林模型（最终锁定的模型口径）。

    为方便部署，仓库中只存放压缩后的 rf_best_model.pkl.bz2；
    若解压后的 pkl 不存在，则首次加载时自动解压。
    """
    if not MODEL_PKL.exists() and MODEL_BZ2.exists():
        with bz2.open(MODEL_BZ2, "rb") as f_in, open(MODEL_PKL, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="正在读取测试集预测结果…")
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(PRED_CSV)


@st.cache_data(show_spinner=False)
def load_locked_metrics() -> dict:
    with open(METRICS_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="正在训练模型（仅使用训练集拟合）…")
def train_models():
    """在训练集上拟合基线 / 逻辑回归 / 随机森林，并计算验证集概率。

    预处理（标准化）放在 Pipeline 内，只在训练集 fit，再应用到验证集，
    符合课程「预处理仅在训练集拟合」的要求。随机森林与组员交付的
    rf_best_model.pkl 采用相同配置（300 棵树，random_state=42）。
    """
    df = load_data()
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "valid"]

    X_tr, y_tr = tr[FEATURES], tr["default_next_month"]
    X_va, y_va = va[FEATURES], va["default_next_month"]

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(X_tr, y_tr)

    return {
        "rf": rf,
        "lr": lr,
        "y_valid": y_va.to_numpy(),
        "proba_valid_rf": rf.predict_proba(X_va)[:, 1],
        "proba_valid_lr": lr.predict_proba(X_va)[:, 1],
    }


# ---------------------------------------------------------------- 通用计算
def pick_threshold(y_true: np.ndarray, proba: np.ndarray,
                   min_recall: float = PAY_STATUS_MIN_RECALL):
    """在验证集上选择报警阈值：满足 召回率 ≥ min_recall 的候选中精确率最高者。

    返回 (阈值, 该阈值下的精确率, 该阈值下的召回率)。
    """
    order = np.argsort(-proba)                     # 按风险概率从高到低排序
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)                       # 前 k 名中真正违约数
    fp = np.cumsum(1 - y_sorted)                   # 前 k 名中误报数
    recall = tp / max(y_true.sum(), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
    ok = np.where(recall >= min_recall)[0]
    if len(ok) == 0:                               # 没有满足约束的候选时取最宽松阈值
        return float(proba[order][-1]), 0.0, 0.0
    best = ok[np.argmax(precision[ok])]
    return float(proba[order][best]), float(precision[best]), float(recall[best])


def pr_curve_points(y_true: np.ndarray, proba: np.ndarray):
    """给阈值-精确率/召回率曲线准备网格点。"""
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    pre, rec = [], []
    for t in grid:
        pred = (proba >= t).astype(int)
        pre.append(precision_score(y_true, pred, zero_division=0))
        rec.append(recall_score(y_true, pred, zero_division=0))
    return grid, np.array(pre), np.array(rec)


def style_fig(fig, x_title, y_title, legend_title=None, height=420):
    """统一图表样式：白底、中文轴名、图例名。"""
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=50, r=20, t=50, b=45),
        legend_title_text=legend_title,
        font=dict(size=13),
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return fig


# ================================================================ 页面 1 数据概览
def page_overview():
    df = load_data()
    st.title("📊 数据概览")
    st.caption("数据来源：01_credit_default.csv（台湾地区 2005 年信用卡客户还款记录，30,000 条；"
               "训练 / 验证 / 测试 = 20,999 / 4,500 / 4,501）")

    # ---------- 统计卡片 ----------
    dup_ids = int(df["record_id"].duplicated().sum())
    n_missing = int(df.isna().sum().sum())
    default_rate = df["default_next_month"].mean()

    n_default = int(df["default_next_month"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("总样本数", f"{len(df):,}")
    c2.metric("总体违约率", f"{default_rate:.2%}")
    c3.metric("违约客户数", f"{n_default:,}")
    c4, c5 = st.columns(2)
    c4.metric("缺失值总数", n_missing)
    c5.metric("record_id 重复数", dup_ids)

    # ---------- 分组样本量表 ----------
    st.subheader("数据划分与各组违约率")
    split_stat = (
        df.groupby("split")
        .agg(样本量=("record_id", "size"),
             违约数=("default_next_month", "sum"),
             违约率=("default_next_month", "mean"))
        .reindex(["train", "valid", "test"])
        .rename(index={"train": "训练集", "valid": "验证集", "test": "测试集"})
        .reset_index()
    )
    split_stat["违约率"] = split_stat["违约率"].map(lambda x: f"{x:.2%}")
    st.dataframe(split_stat, width='stretch', hide_index=True)

    # ---------- 分布图 ----------
    plot_df = df.copy()
    plot_df["违约标签"] = plot_df["default_next_month"].map({0: "正常", 1: "违约"})

    col_a, col_b = st.columns(2)
    with col_a:
        fig_limit = px.histogram(
            plot_df, x="credit_limit_ntd", color="违约标签", nbins=60,
            barmode="overlay", opacity=0.6,
            color_discrete_map={"正常": "#4C78A8", "违约": "#E45756"},
        )
        style_fig(fig_limit, "授信额度（新台币）", "客户数", "分组", height=380)
        st.plotly_chart(fig_limit, width='stretch')
    with col_b:
        fig_age = px.histogram(
            plot_df, x="age_years", color="违约标签", nbins=45,
            barmode="overlay", opacity=0.6,
            color_discrete_map={"正常": "#4C78A8", "违约": "#E45756"},
        )
        style_fig(fig_age, "年龄（岁）", "客户数", "分组", height=380)
        st.plotly_chart(fig_age, width='stretch')

    # 还款状态分布（月份可选，切换后图表自动更新）
    st.subheader("还款状态分布（可切换月份）")
    month = st.selectbox(
        "选择月份",
        PAY_MONTHS,
        index=0,
        format_func=lambda m: PAY_MONTH_CN[m],
    )
    status_col = f"pay_status_{month}"
    status_stat = (
        df.groupby(status_col).size().reset_index(name="客户数")
        .sort_values(status_col)
    )
    status_stat["状态说明"] = status_stat[status_col].map(pay_status_cn)
    fig_status = px.bar(
        status_stat, x=status_col, y="客户数", color="客户数",
        color_continuous_scale="Blues", text_auto=True,
    )
    style_fig(fig_status, "还款状态编码", "客户数", height=380)
    fig_status.update_coloraxes(showscale=False)
    st.plotly_chart(fig_status, width='stretch')

    # ---------- 编码说明 ----------
    with st.expander("📖 字段编码说明（点击展开）"):
        st.markdown(
            """
| 字段 | 编码含义 |
| --- | --- |
| 还款状态 `pay_status_*` | **-2 = 无消费**，**-1 = 全额还款**，**0 = 循环信用**，**1~8 = 逾期月数**。负值是正常编码，**不是缺失值**，不可删除或当作 NaN 处理。 |
| 性别 `sex_code` | 1 = 男，2 = 女 |
| 教育 `education_code` | 1 = 研究生及以上，2 = 本科，3 = 高中，4 = 其他，0/5/6 = 未知 |
| 婚姻 `marriage_code` | 1 = 已婚，2 = 未婚，3 = 其他，0 = 其他 |
| 金额类字段 | 单位为新台币（NTD），含授信额度、各月账单金额与还款金额 |
"""
        )
        st.info("💡 数据无逐笔申请日期，训练/验证/测试为分层随机划分（固定随机种子），"
                "并非时间外推测试；详见课程数据包 README。")


# ================================================================ 页面 2 模型比较
def page_compare():
    st.title("⚖️ 模型比较（验证集）")
    st.caption("基线、逻辑回归、随机森林均只在训练集拟合；模型选择与阈值选择只用验证集，"
               "测试集结果见「最终评价」页。")

    res = train_models()
    y_va = res["y_valid"]

    thr_rf, pre_rf, rec_rf = pick_threshold(y_va, res["proba_valid_rf"])
    thr_lr, pre_lr, rec_lr = pick_threshold(y_va, res["proba_valid_lr"])

    # ---------- 模型切换 ----------
    model_name = st.radio(
        "选择模型查看验证集表现",
        ["基线（全判不违约）", "逻辑回归", "随机森林"],
        horizontal=True,
    )

    if model_name == "基线（全判不违约）":
        st.info("基线将所有客户判为「不违约」，作为模型效果的最低参照。")
        base_pred = np.zeros_like(y_va)
        m1, m2, m3 = st.columns(3)
        m1.metric("验证集准确率", f"{accuracy_score(y_va, base_pred):.2%}")
        m2.metric("平衡准确率", f"{balanced_accuracy_score(y_va, base_pred):.2%}")
        m3.metric("违约类召回率", "0.00%")
        st.warning("基线不预测任何「违约」正类，因此精确率与 ROC-AUC 无法计算，"
                   "界面如实显示原因而不填造数值。")
    else:
        proba = res["proba_valid_rf"] if model_name == "随机森林" else res["proba_valid_lr"]
        auto_thr = thr_rf if model_name == "随机森林" else thr_lr

        auc = roc_auc_score(y_va, proba)
        m1, m2, m3 = st.columns(3)
        m1.metric("验证集 ROC-AUC", f"{auc:.4f}")
        m2.metric("自动选定阈值", f"{auto_thr:.4f}")
        m3.metric("阈值选择规则", "召回率 ≥ 70% 下精确率最高")

        # 阈值手动探索（改变滑块即更新下方全部结果）
        st.subheader("阈值调节（观察精确率 / 召回率随阈值变化）")
        thr = st.slider(
            "报警阈值：预测概率 ≥ 阈值则判定为「违约」", 0.05, 0.95,
            value=float(round(auto_thr, 2)), step=0.01,
        )
        pred = (proba >= thr).astype(int)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("精确率", f"{precision_score(y_va, pred, zero_division=0):.4f}")
        c2.metric("召回率", f"{recall_score(y_va, pred, zero_division=0):.4f}")
        c3.metric("报警人数", f"{int(pred.sum()):,}")
        c4.metric("其中真正违约", f"{int(y_va[pred == 1].sum()):,}")

        # 阈值-指标曲线
        grid, pre_pts, rec_pts = pr_curve_points(y_va, proba)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=grid, y=pre_pts, name="精确率",
                                    line=dict(color="#4C78A8", width=2)))
        fig_pr.add_trace(go.Scatter(x=grid, y=rec_pts, name="召回率",
                                    line=dict(color="#F58518", width=2)))
        fig_pr.add_vline(x=auto_thr, line_dash="dash", line_color="#E45756",
                         annotation_text=f"自动阈值 {auto_thr:.3f}")
        fig_pr.add_vline(x=thr, line_dash="dot", line_color="#54A24B",
                         annotation_text=f"当前滑块 {thr:.2f}")
        style_fig(fig_pr, "报警阈值", "得分（验证集）", "指标", height=380)
        st.plotly_chart(fig_pr, width='stretch')

    # ---------- 三模型汇总对比 ----------
    st.subheader("验证集汇总对比")
    summary = pd.DataFrame(
        {
            "模型": ["基线（全判不违约）", "逻辑回归", "随机森林"],
            "验证集 ROC-AUC": ["—（不输出概率，无法计算）",
                          f"{roc_auc_score(y_va, res['proba_valid_lr']):.4f}",
                          f"{roc_auc_score(y_va, res['proba_valid_rf']):.4f}"],
            "准确率": [f"{accuracy_score(y_va, np.zeros_like(y_va)):.2%}",
                    f"{accuracy_score(y_va, res['proba_valid_lr'] >= thr_lr):.2%}",
                    f"{accuracy_score(y_va, res['proba_valid_rf'] >= thr_rf):.2%}"],
            "自动阈值": ["—", f"{thr_lr:.4f}", f"{thr_rf:.4f}"],
            "阈值下精确率": ["—（不预测正类）", f"{pre_lr:.4f}", f"{pre_rf:.4f}"],
            "阈值下召回率": ["0.00%", f"{rec_lr:.4f}", f"{rec_rf:.4f}"],
        }
    )
    st.dataframe(summary, width='stretch', hide_index=True)

    # ---------- 随机森林特征重要性 ----------
    with st.expander("🌳 随机森林特征重要性 Top 15（点击展开）"):
        imp = res["rf"].feature_importances_
        imp_df = pd.DataFrame({"特征": FEATURES, "重要性": imp}).nlargest(15, "重要性")
        fig_imp = px.bar(imp_df.sort_values("重要性"), x="重要性", y="特征",
                         orientation="h", color_discrete_sequence=["#4C78A8"])
        style_fig(fig_imp, "重要性（基尼不纯度下降）", "特征", height=460)
        st.plotly_chart(fig_imp, width='stretch')


# ================================================================ 页面 3 最终评价
def page_final():
    st.title("✅ 最终评价（锁定模型 · 测试集）")
    st.caption("组员锁定模型（随机森林，300 棵树）与报警阈值后，**对测试集仅评价一次**。"
               "本页结果直接来自组员交付的 test_metrics.json 与 test_predictions.csv，"
               "界面不重复调参、不修改指标。")

    try:
        metrics = load_locked_metrics()
    except FileNotFoundError:
        st.error("未找到 models/test_metrics.json。缺少锁定指标文件时本页无法展示，"
                 "请向负责测试的组员索取，界面不填造数值。")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c2.metric("违约类精确率", f"{metrics['precision']:.4f}")
    c3.metric("违约类召回率", f"{metrics['recall']:.4f}")
    c4, c5 = st.columns(2)
    c4.metric("报警阈值", f"{metrics['threshold']:.4f}")
    c5.metric("测试集样本", f"{metrics['n_test']:,}")

    st.subheader("混淆矩阵（测试集）")
    try:
        pred_df = load_test_predictions()
        y_true, y_pred = pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except FileNotFoundError:
        st.warning("未找到 models/test_predictions.csv，混淆矩阵改用指标文件中的计数展示。")
        tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]

    cm = np.array([[tn, fp], [fn, tp]])
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues", aspect="auto",
        x=["预测：正常(0)", "预测：违约(1)"],
        y=["真实：正常(0)", "真实：违约(1)"],
    )
    style_fig(fig_cm, "预测标签", "真实标签", height=400)
    fig_cm.update_coloraxes(showscale=False)
    st.plotly_chart(fig_cm, width='stretch')

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("漏判人数（FN）", f"{metrics['missed_defaults_fn']:,}", "损失的违约客户",
              delta_color="inverse")
    k2.metric("误判人数（FP）", f"{metrics['false_alarms_fp']:,}", "被误报的正常客户",
              delta_color="inverse")
    k3.metric("正确拦截（TP）", f"{metrics['tp']:,}")
    k4.metric("正确放行（TN）", f"{metrics['tn']:,}")
    st.info("漏判 = 真违约但未报警，可能增加损失；误判 = 正常客户被报警，可能影响正常客户。"
            "阈值在验证集上按「召回率 ≥ 70% 中精确率最高」选定，本页为锁定后的唯一一次测试集评价。")

    st.subheader("ROC 曲线（测试集）")
    fpr, tpr, _ = roc_curve(y_true, pred_df["proba"])
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, name=f"随机森林（AUC = {metrics['roc_auc']:.4f}）",
        line=dict(color="#4C78A8", width=2), mode="lines"))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="随机猜测（AUC = 0.5）",
        line=dict(color="gray", dash="dash"), mode="lines"))
    style_fig(fig_roc, "假正率 FPR", "真正率 TPR", "曲线", height=420)
    st.plotly_chart(fig_roc, width='stretch')


# ================================================================ 页面 4 客户风险演示
def page_demo():
    df = load_data()
    st.title("👤 客户风险演示")
    st.caption("选择一条已有客户记录，系统用组员交付的锁定模型计算违约概率。"
               "结果仅作教学示例，**不作为授信决定**。")

    try:
        model = load_locked_model()
    except FileNotFoundError:
        st.error("未找到 models/rf_best_model.pkl，演示功能需要该模型文件。请向负责模型训练的组员索取。")
        st.stop()

    if "demo_id" not in st.session_state:
        st.session_state.demo_id = 6

    col_in, col_btn, col_rand = st.columns([3, 1.4, 1.4])
    rid = col_in.number_input("客户编号 record_id（1 ~ 30,000）",
                              min_value=1, max_value=30000, step=1,
                              value=int(st.session_state.demo_id))
    if col_btn.button("🔍 查询该客户", width='stretch'):
        st.session_state.demo_id = int(rid)
    if col_rand.button("🎲 随机抽一位", width='stretch'):
        st.session_state.demo_id = int(np.random.randint(1, 30001))
        st.rerun()

    rid = int(st.session_state.demo_id)
    row = df[df["record_id"] == rid]

    if row.empty:
        st.warning(f"编号 {rid} 不存在，请输入 1 ~ 30,000 之间的有效客户编号。")
        st.stop()

    rec = row.iloc[0]

    # ---------- 客户基本信息 ----------
    st.subheader(f"客户 #{rec['record_id']} 主要信息")
    b1, b2, b3 = st.columns(3)
    b1.metric("授信额度", f"{rec['credit_limit_ntd'] / 10000:.1f} 万新台币")
    b2.metric("年龄", f"{rec['age_years']} 岁")
    b3.metric("性别", SEX_CN.get(rec["sex_code"], "未知"))
    b4, b5 = st.columns(2)
    b4.metric("教育", EDU_CN.get(rec["education_code"], "未知"))
    b5.metric("婚姻", MAR_CN.get(rec["marriage_code"], "未知"))

    hist = pd.DataFrame({
        "月份": [PAY_MONTH_CN[m] for m in PAY_MONTHS],
        "还款状态": [pay_status_cn(rec[f"pay_status_{m}"]) for m in PAY_MONTHS],
        "账单金额(NTD)": [f"{rec[f'bill_amount_{m}_ntd']:,.0f}" for m in PAY_MONTHS],
        "还款金额(NTD)": [f"{rec[f'paid_amount_{m}_ntd']:,.0f}" for m in PAY_MONTHS],
    })
    st.dataframe(hist, width='stretch', hide_index=True)

    # ---------- 风险预测（锁定模型） ----------
    proba = float(model.predict_proba(pd.DataFrame([rec[FEATURES]]))[0, 1])
    try:
        threshold = float(load_locked_metrics()["threshold"])
    except FileNotFoundError:
        threshold = 0.2167  # 指标文件缺失时的默认锁定阈值

    st.subheader("风险预测结果")
    r1, r2, r3 = st.columns(3)
    r1.metric("预测违约概率", f"{proba:.2%}")
    r2.metric("报警阈值", f"{threshold:.2%}")
    r3.metric("预测结果", "⚠️ 高风险" if proba >= threshold else "✅ 低风险")

    st.progress(min(proba, 1.0), text=f"违约概率 {proba:.2%}（阈值 {threshold:.2%}）")

    if proba >= threshold:
        st.error(
            f"该客户预测违约概率 {proba:.2%} 高于报警阈值 {threshold:.2%}，"
            "系统提示**建议安排风险核查**（教学演示，非真实授信建议）。"
        )
    else:
        st.success(
            f"该客户预测违约概率 {proba:.2%} 低于报警阈值 {threshold:.2%}，"
            "系统判定为低风险客户（教学演示）。"
        )

    # ---------- 事后真实标签（与预测分开标示） ----------
    if rec["split"] == "test":
        truth = "违约" if rec["default_next_month"] == 1 else "正常"
        correct = (proba >= threshold) == (rec["default_next_month"] == 1)
        with st.container(border=True):
            st.markdown(
                f"**事后信息（仅用于教学对照，预测时模型不可见）**：该记录属于测试集，"
                f"真实标签为 **{truth}**；本次预测{'正确 ✅' if correct else '错误 ❌'}。"
            )
    else:
        st.caption("该记录属于训练集 / 验证集，模型在拟合阶段见过其标签，"
                   "因此不展示「事后真实标签」对照，以免误导。")

    with st.expander("ℹ️ 关于本演示的说明（点击展开）"):
        st.markdown(
            "- 预测模型：组员交付的随机森林（300 棵树，random_state=42），报警阈值取自验证集。\n"
            "- 性别、教育、婚姻属于**敏感属性**，本系统仅用于课堂公平性讨论；模型输出的是统计相关性，"
            "**相关不等于因果**，不能解释为这些属性「导致」违约。\n"
            "- 本演示不输入、不存储任何真实个人数据，全部记录来自课程公开教学数据集。"
        )


# ---------------------------------------------------------------- 应用入口
st.set_page_config(page_title="信用卡违约风险分析系统", page_icon="💳", layout="wide")

st.sidebar.markdown("## 💳 信用卡客户下月违约\n## 风险分析系统")
st.sidebar.caption("计算机金融学导论 · 课程项目一（教学用途）")

# 数据预检：启动时校验课程数据能否导入，失败则给出明确提示
try:
    n_rows = len(load_data())
    st.sidebar.success(f"数据已加载：{n_rows:,} 条客户记录")
except FileNotFoundError:
    st.sidebar.error("未找到 data/01_credit_default.csv，请确认项目文件夹结构完整。")
    st.stop()
except Exception as e:
    st.sidebar.error(f"数据读取失败：{e}")
    st.stop()

# 多页面导航（st.navigation 每次切换渲染全新容器，避免控件残留）
page = st.navigation([
    st.Page(page_overview, title="数据概览", icon="📊", default=True),
    st.Page(page_compare, title="模型比较", icon="⚖️"),
    st.Page(page_final, title="最终评价", icon="✅"),
    st.Page(page_demo, title="客户风险演示", icon="👤"),
])
page.run()

st.sidebar.divider()
st.sidebar.caption("⚠️ 本系统仅作课堂教学分析，**不用于真实授信决策**。"
                   "性别、年龄、教育、婚姻等敏感属性仅用于公平性讨论；"
                   "统计相关性不等于因果关系。")
