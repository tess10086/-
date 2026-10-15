"""信用卡客户下月违约风险分析 —— Streamlit 界面层。

import credit_pipeline 的函数，不重写任何计算逻辑。
四个页面：数据概览 / 模型比较 / 最终评价（测试集）/ 客户风险演示。
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.metrics import roc_auc_score

import artifacts_io as ai
import credit_pipeline as cp

st.set_page_config(page_title="信用卡违约风险分析", page_icon="💳", layout="wide")

MODEL_LABELS = {"baseline": "基线（全判不违约）", "logreg": "逻辑回归", "tree": "树模型"}
MODEL_LABELS_EN = {
    "Baseline_AllNoDefault": "基线（全判不违约）",
    "LogisticRegression": "逻辑回归",
    "RandomForest": "随机森林",
}


# ---------------------------------------------------------------------------
# 数据加载与训练缓存
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_cached(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        # 部署到云端时工作目录不一定是仓库根目录，回退到 app.py 同级目录再试一次
        alt = Path(__file__).resolve().parent / path
        if alt.exists():
            p = alt
    return cp.load_data(p)


def get_train_result(df: pd.DataFrame, exclude_sensitive: bool = False) -> dict:
    """获取训练结果，缓存在 session_state 中。"""
    key = "tr_no_sens" if exclude_sensitive else "tr_normal"
    if key not in st.session_state:
        train_df = df[df[cp.SPLIT_COL] == "train"]
        valid_df = df[df[cp.SPLIT_COL] == "valid"]
        with st.spinner("模型训练中，请稍候..."):
            st.session_state[key] = cp.train_models(
                train_df, valid_df, exclude_sensitive=exclude_sensitive)
    return st.session_state[key]


def predict_single_prob(result: dict, model_name: str, record_df: pd.DataFrame) -> float:
    """用已训练模型预测单条记录的违约概率。"""
    if model_name == "baseline":
        return 0.0
    pipe = result["models"][model_name]["pipe"]
    pay_encoding = result["pay_encoding"]
    exclude_sens = result.get("exclude_sensitive", False)
    rec = cp._prepare_features(record_df, pay_encoding)
    drop_cols = list(cp.EXCLUDE_COLS)
    if exclude_sens:
        drop_cols += [c for c in cp.SENSITIVE_COLS if c in rec.columns]
    X = rec.drop(columns=drop_cols)
    return float(pipe.predict_proba(X)[0, 1])


def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ 设置")
    csv_path = st.text_input("CSV 文件路径", value="01_credit_default.csv")
    if st.button("重新加载数据", width="stretch"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()
    st.divider()
    page = st.radio(
        "页面导航",
        ["数据概览", "模型比较", "最终评价（测试集）", "客户风险演示"],
    )
    st.divider()
    st.caption("队友交付物状态（artifacts/）")
    for name, ok in ai.artifacts_status().items():
        st.caption(("✅ " if ok else "❌ ") + name)
    _c_err = ai.c_bundle_error()
    if _c_err:
        st.caption(f"⚠️ C 模型加载失败：{_c_err}")

# ---------------------------------------------------------------------------
# 顶部标题 + 免责声明
# ---------------------------------------------------------------------------

st.title("信用卡客户下月违约风险分析")
st.warning("⚠️ 本项目仅作教学分析，不用于真实授信决策。")

# 加载数据
try:
    df = load_cached(csv_path)
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()


# ===========================================================================
# 页面一：数据概览
# ===========================================================================

def page_overview(df: pd.DataFrame) -> None:
    st.header("📊 数据概览")
    audit = cp.audit_data(df)
    total = len(df)
    overall_rate = df[cp.LABEL].mean()

    # 统计卡片
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总样本量", f"{total:,}")
    c2.metric("训练集", f"{audit['train']['n']:,}")
    c3.metric("验证集", f"{audit['valid']['n']:,}")
    c4.metric("测试集", f"{audit['test']['n']:,}")
    c5.metric("整体违约率", f"{overall_rate:.4f}")

    # 按 split 分组统计表
    st.subheader("按划分分组的统计表")
    rows = []
    for name in ["train", "valid", "test"]:
        info = audit[name]
        rows.append({
            "划分": name, "样本量": info["n"], "违约人数": info["default_n"],
            "违约率": info["default_rate"], "缺失值个数": info["missing_total"],
            "record_id重复数": info["record_id_dup"],
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # 图1：授信额度分布
    st.subheader("授信额度分布")
    fig = px.histogram(
        df, x=cp.COL_LIMIT_BAL, color=cp.SPLIT_COL, barmode="overlay", opacity=0.7,
        title=f"授信额度分布（全部 n={total}）",
        labels={cp.COL_LIMIT_BAL: "授信额度（新台币）", cp.SPLIT_COL: "划分"})
    fig.update_layout(xaxis_title="授信额度", yaxis_title="客户数")
    st.plotly_chart(fig, width="stretch")

    # 图2：年龄分布
    st.subheader("年龄分布")
    fig = px.histogram(
        df, x=cp.COL_AGE, color=cp.SPLIT_COL, barmode="overlay", opacity=0.7,
        title=f"年龄分布（全部 n={total}）",
        labels={cp.COL_AGE: "年龄", cp.SPLIT_COL: "划分"})
    fig.update_layout(xaxis_title="年龄", yaxis_title="客户数")
    st.plotly_chart(fig, width="stretch")

    # 图3：最近月还款状态分布
    st.subheader("最近月还款状态 PAY_0 取值分布")
    pay_dist = cp.dist_pay_status(df)
    fig = px.bar(
        pay_dist, x="取值", y="客户数",
        title=f"最近月还款状态取值分布（全部 n={total}）")
    fig.update_layout(xaxis_title="还款状态取值", yaxis_title="客户数")
    st.plotly_chart(fig, width="stretch")

    # 图4：额度使用率分箱 vs 违约率
    st.subheader("额度使用率分箱 vs 违约率")
    df_f = cp.add_derived_features(df)
    df_f["util_bin"] = pd.cut(
        df_f["utilization_1"],
        bins=[-np.inf, 0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf],
        labels=["≤0", "0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0", ">1.0"])
    util_rate = cp.default_rate_by_group(df_f, "util_bin")
    fig = px.bar(
        util_rate, x="组值", y="违约率",
        title=f"额度使用率分箱 vs 违约率（全部 n={total}）")
    fig.update_layout(xaxis_title="额度使用率区间", yaxis_title="违约率")
    st.plotly_chart(fig, width="stretch")

    # 还款状态编码含义
    st.subheader("还款状态编码含义")
    st.markdown("**负值是有效编码，不作为缺失值处理。**")
    st.dataframe(cp.describe_pay_codes(df), width="stretch", hide_index=True)

    st.info("数据无逐笔申请日期，划分为分层随机划分，结论不可外推到未来时段。")


# ===========================================================================
# 页面二：模型比较
# ===========================================================================

def page_model_compare(df: pd.DataFrame) -> None:
    st.header("🔍 模型比较")

    # ---- C 同学交付：团队正式口径的模型比较（读 artifacts，页面不重算）----
    st.subheader("C 同学交付：验证集模型比较")
    c_thr = ai.load_c_threshold()
    mc_valid = ai.load_c_model_comparison("valid")
    if c_thr is None and mc_valid is None:
        st.warning("未找到 C 同学的交付物（artifacts/model_comparison.csv、threshold.json）。")
    else:
        if c_thr:
            st.info(
                f"C 同学选定模型：**{MODEL_LABELS_EN.get(c_thr.get('model_name'), c_thr.get('model_name'))}**，"
                f"阈值 **{c_thr.get('threshold')}**；依据：{c_thr.get('selection_reason', '-')}")
        if mc_valid is not None:
            show = mc_valid.copy()
            show["model"] = show["model"].map(lambda x: MODEL_LABELS_EN.get(x, x))
            st.dataframe(show, width="stretch", hide_index=True)
            st.caption("来源：artifacts/model_comparison.csv（split=valid，由 C 同学计算）")
        tc = ai.load_c_threshold_curve(c_thr.get("model_name") if c_thr else "RandomForest")
        if tc is not None:
            fig = go.Figure()
            for col, lab in [("precision", "精确率"), ("recall", "召回率"), ("f1", "F1")]:
                fig.add_trace(go.Scatter(x=tc["threshold"], y=tc[col], name=lab, mode="lines"))
            if c_thr:
                fig.add_vline(x=c_thr.get("threshold"), line_dash="dash", line_color="red",
                              annotation_text=f"选定阈值={c_thr.get('threshold')}")
            fig.update_layout(title="阈值—指标曲线（C 同学，验证集）",
                              xaxis_title="判定阈值", yaxis_title="指标值")
            st.plotly_chart(fig, width="stretch")
            st.caption("来源：artifacts/threshold_comparison.csv")
        fi = ai.load_c_feature_importance(20)
        if fi is not None:
            fig = px.bar(
                fi.sort_values("importance_or_coefficient"),
                x="importance_or_coefficient", y="feature", orientation="h",
                title="特征重要性 Top20（C 同学的随机森林）")
            fig.update_layout(xaxis_title="重要性", yaxis_title="特征（编码后）")
            st.plotly_chart(fig, width="stretch")
            st.caption("来源：artifacts/feature_importance.csv")

    st.divider()
    st.subheader("界面在线复算（E 的 Streamlit 现场训练）")
    st.caption("此处的模型由界面在 train 上现场拟合，参数与 C 同学的随机森林不同，指标会有小差异；"
               "团队正式口径以上方 C 同学的交付物为准。")

    locked = cp.load_locked()
    is_locked = locked is not None
    result = get_train_result(df)
    y_valid = result["y_valid"]

    # 锁定状态下使用锁定的模型与阈值
    if is_locked:
        model_name = locked["model_name"]
        threshold = locked["threshold"]
        st.info(
            f"已锁定：模型={MODEL_LABELS.get(model_name, model_name)}，阈值={threshold}。"
            f"如需更改请先在「最终评价」页点击重置（将清空测试结果）。")
    else:
        # 模型切换
        model_name = st.selectbox(
            "选择模型", ["baseline", "logreg", "tree"], index=1,
            format_func=lambda x: MODEL_LABELS[x])

        # 阈值选择
        st.subheader("阈值选择（仅用验证集）")
        criterion = st.radio("选择准则", ["max_f1", "recall_ge", "min_cost"], horizontal=True)
        recall_target = 0.60
        cost_ratio = (5, 1)
        if criterion == "recall_ge":
            recall_target = st.slider("目标召回率", 0.30, 0.95, 0.60, 0.05)
        elif criterion == "min_cost":
            fn_cost = st.slider("漏判成本（误判成本固定为1）", 1, 20, 5)
            cost_ratio = (fn_cost, 1)

    y_prob = result["models"][model_name]["y_prob_valid"]
    sweep = cp.threshold_sweep(y_valid, y_prob)

    if not is_locked:
        threshold, reason = cp.pick_threshold(
            sweep, criterion, recall_target=recall_target, cost_ratio=cost_ratio)
        if threshold is None:
            # 基线（全判不违约）预测无正例，精确率/F1 均无法计算，阈值无效
            st.warning(
                f"当前模型「{MODEL_LABELS[model_name]}」无法选出有意义阈值：{reason}。"
                f"已临时按 0.5 展示，请在上方切换到「逻辑回归」或「树模型」后再选阈值。")
            threshold = 0.5
        else:
            st.markdown(f"**选中阈值 {threshold}**，依据：{reason}")
    else:
        # 锁定后仍展示阈值曲线，但不允许改阈值
        threshold_val = locked["threshold"]
        st.markdown(f"**锁定阈值 {threshold_val}**")

    # 验证集指标：阈值 0.5 与选定阈值对比
    st.subheader("验证集指标对比")
    m_05 = cp.compute_metrics(y_valid, y_prob, 0.5)
    m_sel = cp.compute_metrics(y_valid, y_prob, threshold)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**阈值 0.5**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", _fmt(m_05["roc_auc"]))
        c2.metric("精确率", _fmt(m_05["precision"]))
        c3.metric("召回率", _fmt(m_05["recall"]))
        c4.metric("F1", _fmt(m_05["f1"]))
        if m_05.get("reason"):
            st.caption(f"原因：{m_05['reason']}")
    with col2:
        st.markdown(f"**选定阈值 {threshold}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", _fmt(m_sel["roc_auc"]))
        c2.metric("精确率", _fmt(m_sel["precision"]))
        c3.metric("召回率", _fmt(m_sel["recall"]))
        c4.metric("F1", _fmt(m_sel["f1"]))
        if m_sel.get("reason"):
            st.caption(f"原因：{m_sel['reason']}")

    # 阈值-指标曲线
    st.subheader("阈值—指标曲线（验证集）")
    fig = go.Figure()
    for col, label in [("precision", "精确率"), ("recall", "召回率"), ("f1", "F1")]:
        fig.add_trace(go.Scatter(
            x=sweep["threshold"], y=sweep[col], name=label, mode="lines"))
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"阈值={threshold}")
    fig.update_layout(
        title=f"阈值—精确率/召回率/F1 曲线（验证集 n={len(y_valid)}）",
        xaxis_title="判定阈值", yaxis_title="指标值")
    st.plotly_chart(fig, width="stretch")

    # 三模型对比表
    st.subheader("三模型验证集对比（阈值 0.5）")
    cmp_rows = []
    for name in ["baseline", "logreg", "tree"]:
        yp = result["models"][name]["y_prob_valid"]
        mm = cp.compute_metrics(y_valid, yp, 0.5)
        cmp_rows.append({
            "模型": MODEL_LABELS[name], "AUC": mm["roc_auc"],
            "精确率": mm["precision"], "召回率": mm["recall"], "F1": mm["f1"],
        })
    cmp_df = pd.DataFrame(cmp_rows)

    def _highlight(row):
        if row["模型"] == MODEL_LABELS[model_name]:
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(cmp_df.style.apply(_highlight, axis=1),
                 width="stretch", hide_index=True)

    # 锁定按钮
    if not is_locked:
        st.subheader("锁定模型与阈值")
        if st.button("🔒 锁定模型与阈值", type="primary"):
            vm = cp.compute_metrics(y_valid, y_prob, threshold)
            vm["pay_encoding"] = result["pay_encoding"]
            vm["exclude_sensitive"] = result.get("exclude_sensitive", False)
            cp.lock_model(model_name, threshold, vm)
            st.success(f"已锁定：模型={MODEL_LABELS[model_name]}，阈值={threshold}")
            st.rerun()
    else:
        st.caption("已锁定，如需更改请先在「最终评价」页点击重置（将清空测试结果）。")


# ===========================================================================
# 页面三：最终评价（测试集）
# ===========================================================================

def page_final_evaluate(df: pd.DataFrame) -> None:
    st.header("🏁 最终评价（测试集）")

    # ---- D 同学交付：锁定的测试集最终结果（只评价一次，页面不重算）----
    d = ai.load_d_test_result()
    m_d = ai.d_result_to_metrics(d) if d else None
    if d is not None:
        st.success("D 同学锁定的测试集最终结果（artifacts/test_result.json，只评价一次，本页直接读文件不重算）")
        st.markdown(
            f"**模型**：{MODEL_LABELS_EN.get(d.get('model_name'), d.get('model_name'))}　"
            f"**阈值**：{d.get('threshold')}（来源：{d.get('threshold_source')}）　"
            f"**测试集样本量**：n={m_d['n_test']}")
        st.markdown(
            f"**评价时间**：{d.get('evaluated_at')}　|　**数据指纹**：`{d.get('data_fingerprint')}`　|　"
            f"**代码版本**：{d.get('code_version')}")
        if not d.get("feature_order_verified", True):
            st.warning(f"⚠️ D 同学标注 feature_order_verified=false。原文说明：{d.get('feature_note')}")
        if d.get("unavailable_reasons"):
            st.info(f"不可计算的指标：{d['unavailable_reasons']}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC-AUC", _fmt(m_d["roc_auc"]))
        c2.metric("违约类精确率", _fmt(m_d["precision"]))
        c3.metric("召回率", _fmt(m_d["recall"]))
        c4.metric("F1", _fmt(m_d["f1"]))
        c5.metric("报警率", _fmt(m_d["alarm_rate"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("误判人数（FP）", m_d["误判人数"])
        c2.metric("漏判人数（FN）", m_d["漏判人数"])
        c3.metric("准确率", _fmt(m_d["accuracy"]))

        cm = [[m_d["TN"], m_d["FP"]], [m_d["FN"], m_d["TP"]]]
        fig = px.imshow(
            cm, text_auto=True,
            x=["预测不违约(0)", "预测违约(1)"],
            y=["实际不违约(0)", "实际违约(1)"],
            title=f"混淆矩阵（测试集 n={m_d['n_test']}，D 同学交付）",
            color_continuous_scale="Blues")
        fig.update_layout(xaxis_title="预测", yaxis_title="实际")
        st.plotly_chart(fig, width="stretch")
        st.caption("正类=违约。测试集仅评价一次，不得反复调参后当作独立评估。")
    else:
        st.warning("未找到 artifacts/test_result.json，请向 D 同学索取测试集评价结果。")

    # ---- 在线复算核对：用 C 同学的模型在测试集上复算，验证与 D 一致 ----
    with st.expander("在线复算核对（用 C 同学的模型在测试集上复算，验证与 D 同学结果一致）"):
        bundle = ai.load_c_bundle()
        if bundle is None:
            st.warning(f"C 同学模型不可用：{ai.c_bundle_error()}")
        else:
            test_df = df[df[cp.SPLIT_COL] == "test"]
            y_true, y_prob = ai.predict_test_with_c_model(bundle, test_df)
            thr = float(bundle["threshold"])
            yp = (y_prob >= thr).astype(int)
            tp = int(((yp == 1) & (y_true == 1)).sum())
            fp = int(((yp == 1) & (y_true == 0)).sum())
            fn = int(((yp == 0) & (y_true == 1)).sum())
            tn = int(((yp == 0) & (y_true == 0)).sum())
            row = {
                "ROC-AUC": round(roc_auc_score(y_true, y_prob), 6),
                "精确率": round(tp / max(tp + fp, 1), 6),
                "召回率": round(tp / max(tp + fn, 1), 6),
                "TN/FP/FN/TP": f"{tn}/{fp}/{fn}/{tp}",
            }
            st.dataframe(pd.DataFrame([row]), width="stretch", hide_index=True)
            if m_d is not None:
                same = (abs(row["ROC-AUC"] - (m_d["roc_auc"] or 0)) < 1e-6
                        and (tn, fp, fn, tp) == (m_d["TN"], m_d["FP"], m_d["FN"], m_d["TP"]))
                if same:
                    st.success("复算结果与 D 同学的 test_result.json 完全一致 ✅")
                else:
                    st.warning("复算结果与 D 同学不一致，请核对特征列顺序、阈值来源与数据版本。")

    st.divider()
    st.subheader("界面现场训练流程（E 的交互式选型与锁定）")
    locked = cp.load_locked()
    if locked is None:
        st.warning("尚未锁定模型，请先在「模型比较」页锁定模型与阈值。")
        return

    train_df = df[df[cp.SPLIT_COL] == "train"]
    test_df = df[df[cp.SPLIT_COL] == "test"]

    with st.spinner("测试集评价中..."):
        result = cp.final_evaluate(locked, train_df, test_df)

    m = result["metrics"]
    bm = result["baseline_metrics"]

    st.markdown(
        f"**锁定模型**：{MODEL_LABELS.get(result['model_name'], result['model_name'])}  "
        f"**阈值**：{result['threshold']}  **测试集样本量**：n={result['n_test']}")
    st.markdown(f"**评价时间**：{result['evaluate_time']}")
    st.info(result["note"])

    # 指标卡片
    st.subheader("测试集最终指标（正类=违约）")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", _fmt(m["roc_auc"]))
    c2.metric("精确率", _fmt(m["precision"]))
    c3.metric("召回率", _fmt(m["recall"]))
    c4.metric("F1", _fmt(m["f1"]))
    c5.metric("报警率", _fmt(m["alarm_rate"]))
    if m.get("reason"):
        st.caption(f"指标原因：{m['reason']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("误判人数（FP）", m["误判人数"])
    c2.metric("漏判人数（FN）", m["漏判人数"])
    c3.metric("准确率", _fmt(m["accuracy"]))

    # 混淆矩阵热力图
    st.subheader("混淆矩阵")
    cm = [[m["TN"], m["FP"]], [m["FN"], m["TP"]]]
    fig = px.imshow(
        cm, text_auto=True,
        x=["预测不违约(0)", "预测违约(1)"],
        y=["实际不违约(0)", "实际违约(1)"],
        title=f"混淆矩阵（测试集 n={result['n_test']}）",
        color_continuous_scale="Blues")
    fig.update_layout(xaxis_title="预测", yaxis_title="实际")
    st.plotly_chart(fig, width="stretch")

    # 基线对照
    st.subheader("基线（全判不违约）对照")
    bl_rows = [{
        "": "锁定模型", "AUC": m["roc_auc"], "精确率": m["precision"],
        "召回率": m["recall"], "F1": m["f1"], "误判": m["误判人数"], "漏判": m["漏判人数"],
    }, {
        "": "基线", "AUC": bm["roc_auc"], "精确率": bm["precision"],
        "召回率": bm["recall"], "F1": bm["f1"], "误判": bm["误判人数"], "漏判": bm["漏判人数"],
    }]
    st.dataframe(pd.DataFrame(bl_rows), width="stretch", hide_index=True)
    if bm.get("reason"):
        st.caption(f"基线原因：{bm['reason']}")

    # 重置按钮（二次确认）
    st.subheader("重置")
    if st.button("重置锁定（将删除测试结果并允许重新选型）"):
        if st.checkbox("我确认要重置"):
            cp.reset_evaluation()
            st.session_state.clear()
            st.success("已重置，请重新在「模型比较」页选型与锁定。")
            st.rerun()


# ===========================================================================
# 页面四：客户风险演示
# ===========================================================================

def page_customer_demo(df: pd.DataFrame) -> None:
    st.header("👤 客户风险演示")

    # 排除敏感属性开关
    exclude_sensitive = st.checkbox("排除敏感属性重新训练（将重置锁定与测试结果）")
    if exclude_sensitive:
        st.warning("此项会重置锁定与测试结果，需重新选型与评价。")
        if not st.session_state.get("sens_reset"):
            cp.reset_evaluation()
            st.session_state["sens_reset"] = True
    else:
        st.session_state["sens_reset"] = False

    # 模型来源：优先用 C 同学交付的正式模型
    bundle = ai.load_c_bundle()
    if bundle is not None:
        source = st.radio(
            "使用哪个模型预测",
            ["C 同学交付模型（团队正式口径）", "界面在线训练模型（现场复算）"],
            horizontal=True)
    else:
        st.warning(f"C 同学模型不可用（{ai.c_bundle_error()}），本页只能用界面在线训练模型。")
        source = "界面在线训练模型（现场复算）"
    use_c_model = source.startswith("C 同学")

    # 确定使用的模型与阈值
    result = None
    if use_c_model:
        model_name = "c_model"
        threshold = float(bundle["threshold"])
        st.info(f"使用 C 同学交付的 {bundle['model_name']}，阈值={threshold}"
                f"（选定值来自 artifacts/threshold.json）")
    else:
        result = get_train_result(df, exclude_sensitive=exclude_sensitive)
        locked = cp.load_locked()
        if locked and not exclude_sensitive:
            model_name = locked["model_name"]
            threshold = locked["threshold"]
            st.info(f"使用锁定模型：{MODEL_LABELS.get(model_name, model_name)}，阈值={threshold}")
        else:
            model_name = st.selectbox(
                "选择模型", ["baseline", "logreg", "tree"],
                format_func=lambda x: MODEL_LABELS[x])
            threshold = st.slider("判定阈值", 0.01, 0.99, 0.24, 0.01)

    # 取数方式
    st.subheader("选择客户")
    method = st.radio("取数方式", ["输入 record_id", "从测试集随机抽取"], horizontal=True)

    record = None
    if method == "输入 record_id":
        rid = st.number_input("record_id", min_value=1, value=1, step=1)
        if st.button("查询"):
            row = df[df["record_id"] == int(rid)]
            if row.empty:
                st.warning(f"record_id={int(rid)} 不存在，请确认编号。")
            else:
                record = row.iloc[0]
    else:
        if st.button("🎲 随机抽取一条测试集记录"):
            test_df = df[df[cp.SPLIT_COL] == "test"]
            record = test_df.sample(1, random_state=None).iloc[0]
            st.session_state["demo_record"] = record
        if "demo_record" in st.session_state:
            record = st.session_state["demo_record"]

    if record is None:
        st.caption("请选择客户后查看风险演示。")
        _show_sensitive_notice()
        return

    # 预测
    record_df = pd.DataFrame([record])
    if use_c_model:
        prob = ai.predict_with_c_model(bundle, record_df)
    else:
        prob = predict_single_prob(result, model_name, record_df)

    # 模型预测结果区
    st.subheader("模型预测结果")
    st.write(f"**风险概率**：{prob:.4f}")
    st.progress(min(float(prob), 1.0))
    st.write(f"**当前阈值**：{threshold}")

    if prob >= threshold:
        st.error(f"判定结果：⚠️ 报警（预测违约），风险概率 {prob:.4f} ≥ 阈值 {threshold}")
    else:
        st.success(f"判定结果：✅ 不报警（预测正常），风险概率 {prob:.4f} < 阈值 {threshold}")

    # 主要输入信息
    st.subheader("主要输入信息")
    info_cols = ["record_id", cp.COL_LIMIT_BAL, cp.COL_SEX, cp.COL_EDUCATION,
                 cp.COL_MARRIAGE, cp.COL_AGE, cp.PAY_COLS[0],
                 cp.BILL_COLS[0], cp.PAY_AMT_COLS[0]]
    info_cols = [c for c in info_cols if c in record_df.columns]
    rec_feat = cp.add_derived_features(record_df)
    info = record_df[info_cols].copy()
    info["额度使用率"] = rec_feat["utilization_1"].values
    info["还款占账单比"] = rec_feat["pay_ratio_1"].values
    info["最大逾期月数"] = rec_feat["max_delinq"].values
    info["逾期月数个数"] = rec_feat["n_delinq_months"].values
    st.dataframe(info.T, width="stretch")

    # 事后真实标签（分区显示）
    if record[cp.SPLIT_COL] == "test":
        st.subheader("事后真实标签（仅用于对照，不参与预测）")
        true_label = int(record[cp.LABEL])
        if true_label == 1:
            st.markdown(f"**真实标签**：{true_label}（违约）")
        else:
            st.markdown(f"**真实标签**：{true_label}（正常）")
        st.caption("此标签为事后结果，仅用于评估模型预测准确性，不得用于调整任何参数。")
    else:
        st.caption(f"该记录来自 {record[cp.SPLIT_COL]} 集，无事后标签展示。")

    _show_sensitive_notice()


def _show_sensitive_notice() -> None:
    """底部固定的敏感属性说明。"""
    st.divider()
    st.caption("本结果仅为教学示例，不构成授信决定。")
    st.caption(
        "模型中使用的性别、婚姻、教育、年龄等属于敏感属性；"
        "相关性不等于因果关系，模型输出不得作为对任何个人或群体的差别对待依据。")


# ---------------------------------------------------------------------------
# 页面分发
# ---------------------------------------------------------------------------

if page == "数据概览":
    page_overview(df)
elif page == "模型比较":
    page_model_compare(df)
elif page == "最终评价（测试集）":
    page_final_evaluate(df)
elif page == "客户风险演示":
    page_customer_demo(df)