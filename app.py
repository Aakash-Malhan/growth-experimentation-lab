import gradio as gr
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import io

# Synthetic data generator

@dataclass
class SimConfig:
    n_users: int = 50000
    n_days: int = 30
    ab_split: float = 0.5
    conv_base: float = 0.08
    conv_lift_b: float = 0.02
    ctr_base: float = 0.18
    ctr_lift_b: float = 0.03
    session_mu: float = 3.5
    dur_mu: float = 180.0
    dur_sigma: float = 60.0
    seed: int = 42

def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def simulate_data(cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    users = pd.DataFrame({
        "user_id": np.arange(cfg.n_users, dtype=int),
        "latent_engagement": rng.normal(0, 1, cfg.n_users),
        "pre_exp_baseline": rng.normal(1.0, 0.2, cfg.n_users)  # proxy covariate
    })

    users["variant"] = np.where(rng.uniform(0, 1, cfg.n_users) > (1 - cfg.ab_split), "B", "A")

    lam = cfg.session_mu * np.exp(0.15 * users["latent_engagement"])
    users["sessions"] = rng.poisson(lam=lam).clip(0).astype(int)

    rows = []
    if users["sessions"].sum() > 0:
        for uid, var, sess in users[["user_id", "variant", "sessions"]].itertuples(index=False):
            if sess == 0: continue
            days = rng.integers(0, 30, size=int(sess))
            rows.append(pd.DataFrame({"user_id": uid, "variant": var, "day": days}))
    sessions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["user_id","variant","day"])

    if not sessions.empty:
        p_ctr = np.where(sessions["variant"].eq("B"), cfg.ctr_base + cfg.ctr_lift_b, cfg.ctr_base)
        p_ctr = np.clip(p_ctr, 0, 1)
        sessions["clicked"] = (rng.uniform(0, 1, len(sessions)) < p_ctr).astype(int)
        sessions["duration_s"] = np.maximum(1, rng.normal(cfg.dur_mu, cfg.dur_sigma, len(sessions)))
    else:
        sessions["clicked"] = pd.Series(dtype=int)
        sessions["duration_s"] = pd.Series(dtype=float)

    base_logit = _logit(cfg.conv_base)
    conv_logit = base_logit + 0.60 * users["latent_engagement"] + 0.40 * (users["pre_exp_baseline"] - 1.0)
    p_conv = _sigmoid(conv_logit)
    p_conv = np.where(users["variant"].eq("B"), np.clip(p_conv + cfg.conv_lift_b, 0, 1), p_conv)
    users["converted"] = (rng.uniform(0, 1, cfg.n_users) < p_conv).astype(int)

    clicks_by_user = sessions.groupby("user_id")["clicked"].max() if not sessions.empty else pd.Series(dtype=int)
    users["viewed"] = (users["sessions"] > 0).astype(int)
    users["clicked_any"] = users["user_id"].map(clicks_by_user).fillna(0).astype(int)
    users["signed_up"] = users["converted"].astype(int)

    if not sessions.empty:
        daily_active = sessions.groupby(["variant","day"])["user_id"].nunique().reset_index(name="active_users")
    else:
        daily_active = pd.DataFrame(columns=["variant","day","active_users"])
    pop = users.groupby("variant")["user_id"].nunique().reset_index(name="total_users")
    retention = daily_active.merge(pop, on="variant", how="left")
    retention["active"] = np.where(retention.get("total_users",0).eq(0), 0, retention["active_users"]/retention["total_users"])
    return users, sessions, retention[["variant","day","active"]].copy()


# Helper metrics

def ab_summary(users: pd.DataFrame) -> pd.DataFrame:
    grp = users.groupby("variant", as_index=False).agg(
        n_users=("user_id", "count"),
        viewed=("viewed", "mean"),
        clicked_any=("clicked_any", "mean"),
        conversion=("signed_up", "mean"),
    )
    for c in ["viewed","clicked_any","conversion"]:
        grp[c] = (100*grp[c]).round(2)
    return grp

def lift_confint(p1, n1, p2, n2, n_boot=3000, seed=7):
    rng = np.random.default_rng(seed)
    lift = p2 - p1
    p_pool = (p1*n1 + p2*n2) / max(n1+n2, 1)
    se = np.sqrt(max(p_pool*(1-p_pool),1e-9) * (1/n1 + 1/n2))
    z = (p2 - p1) / (se + 1e-12)
    p_val = 2*(1 - stats.norm.cdf(abs(z)))
    boot = []
    for _ in range(n_boot):
        a = rng.binomial(1, p1, n1)
        b = rng.binomial(1, p2, n2)
        boot.append(b.mean() - a.mean())
    lo, hi = np.quantile(boot, [0.025, 0.975]).tolist()
    return float(lift), (float(lo), float(hi)), float(p_val)

def cuped_adjustment(y: np.ndarray, x: np.ndarray):
    x_centered = x - x.mean()
    theta = np.cov(y, x_centered, bias=True)[0,1] / (np.var(x_centered)+1e-12)
    y_adj = y - theta * x_centered
    var_orig, var_adj = np.var(y), np.var(y_adj)
    vr_pct = max(0.0, (1.0 - var_adj/(var_orig+1e-12))*100.0)
    return y_adj, float(vr_pct)

def retention_curve_plot(retention: pd.DataFrame) -> go.Figure:
    if retention is None or retention.empty: return go.Figure()
    agg = retention.groupby(["variant","day"])["active"].mean().reset_index()
    fig = px.line(agg, x="day", y="active", color="variant", markers=True,
                  title="Daily Retention (Active Users Share)")
    fig.update_yaxes(tickformat=".0%")
    return fig

def funnel_plot(users: pd.DataFrame) -> go.Figure:
    if users is None or users.empty: return go.Figure()
    rows = []
    for v in ["A","B"]:
        df = users[users["variant"]==v]
        if df.empty: continue
        rows += [
            {"variant":v, "stage":"Viewed", "pct":100*df["viewed"].mean()},
            {"variant":v, "stage":"Clicked","pct":100*df["clicked_any"].mean()},
            {"variant":v, "stage":"Signed Up","pct":100*df["signed_up"].mean()},
        ]
    if not rows: return go.Figure()
    fdf = pd.DataFrame(rows)
    return px.funnel(fdf, x="pct", y="stage", color="variant", title="Funnel Conversion (%)")


# Real data: CookieCats
# =====================
def load_cookie_cats_csv(file_obj: io.BytesIO) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Expected columns: userid, version (gate_30/gate_40), sum_gamerounds, retention_1, retention_7
    We map:
      - variant: gate_30 -> A, gate_40 -> B
      - sessions: sum_gamerounds (proxy)
      - conversion metric chosen later (ret_1 or ret_7)
    """
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        return pd.DataFrame(), f"CSV read failed: {e}"

    needed = {"userid","version","sum_gamerounds","retention_1","retention_7"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame(), f"Missing columns. Need {needed}, found {set(df.columns)}"

    users = pd.DataFrame({
        "user_id": df["userid"].astype("int64"),
        "variant": df["version"].map({"gate_30":"A","gate_40":"B"}).fillna("A"),
        "sessions": df["sum_gamerounds"].astype("int64"),
        # Use gamerounds as a (weak) CUPED covariate proxy; note: in strict setups, a true pre-experiment covariate is preferred.
        "pre_exp_baseline": (df["sum_gamerounds"] + 1).astype(float).pow(0.5) / 10.0,
        "latent_engagement": 0.0  # not used for real data
    })

    # clicks/signups proxies for funnel tab (optional)
    users["viewed"] = (users["sessions"] > 0).astype(int)
    users["clicked_any"] = (users["sessions"] > users["sessions"].median()).astype(int)  # rough proxy
    # signed_up will be set later from chosen retention metric
    users["signed_up"] = 0

    # No session-level table available in dataset; we create empty sessions and retention curves (N/A)
    sessions = pd.DataFrame(columns=["user_id","variant","day","clicked","duration_s"])
    retention = pd.DataFrame(columns=["variant","day","active"])
    # Attach raw retention columns for later selection
    users["retention_1"] = df["retention_1"].astype(int)
    users["retention_7"] = df["retention_7"].astype(int)
    return users, None


# Churn modeling (real/synth)
# =====================
def build_churn_labels_real_cookie(users: pd.DataFrame) -> pd.Series:
    # Define churn as not retained on day 7
    y = (1 - users["retention_7"]).astype(int)
    return y

def churn_model(users: pd.DataFrame, sessions: pd.DataFrame, n_days: int, seed: int = 7, real_cookie=False) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    if real_cookie:
        y = build_churn_labels_real_cookie(users)
        # Feature set from real columns
        X = pd.DataFrame({
            "sessions": users["sessions"],
            "viewed": users["viewed"],
            "clicked_any": users["clicked_any"],
            "pre_exp_baseline": users["pre_exp_baseline"],
            "retention_1": users["retention_1"]
        })
    else:
        # synthetic mode (same as before, with noise)
        if sessions is None:
            y = pd.Series(np.ones(len(users), dtype=int), index=users.index)
        else:
            last_day = sessions.groupby("user_id")["day"].max()
            last_day = users["user_id"].map(last_day).fillna(-1).astype(int)
            no_session_last_7 = last_day < (n_days - 7)
            low_eng = users["sessions"] < users["sessions"].median()
            cold = (low_eng & (users["clicked_any"] == 0) & (users["signed_up"] == 0))
            y = (no_session_last_7 | cold).astype(int)
            noise = rng.uniform(0, 1, len(y)) < 0.10
            y[noise] = 1 - y[noise]

        X = pd.DataFrame({
            "sessions": users["sessions"],
            "viewed": users["viewed"],
            "clicked_any": users["clicked_any"],
            "signed_up": users["signed_up"],
            "pre_exp_baseline": users["pre_exp_baseline"],
            "noise1": rng.normal(0, 1, len(users)),
            "noise2": rng.normal(0, 1, len(users)),
        })

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
    model = LogisticRegression(max_iter=300, C=0.6, class_weight="balanced")
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.40).astype(int)
    auc = roc_auc_score(y_test, proba)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    coef = pd.DataFrame({"feature": X.columns, "coef": model.coef_[0]}).sort_values("coef", ascending=False)
    return {"auc": float(auc), "precision": float(p), "recall": float(r), "f1": float(f1), "coef_df": coef}

# ==============
# Gradio UI
# ==============
with gr.Blocks(title="Growth & Experimentation Lab (Synthetic + Real Data)") as demo:
    gr.Markdown(
        "# Growth & Experimentation Lab\n"
        "A/B testing with CUPED, retention/funnel analytics, and a churn model.\n\n"
        "**Data source options:** Synthetic (full features) or **Real: Cookie Cats CSV** (true A/B on retention)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            data_source = gr.Dropdown(choices=["Synthetic (built-in)", "Real: Cookie Cats CSV"], value="Synthetic (built-in)", label="Data Source")
            cookie_file = gr.File(label="Upload Cookie Cats CSV (when using Real mode)", file_types=[".csv"], visible=True)
            metric_choice = gr.Radio(choices=["Retention (1-day)", "Retention (7-day)"], value="Retention (7-day)", label="Conversion metric (Real mode)")

            # Synthetic controls
            n_users = gr.Slider(5000, 200000, value=50000, step=5000, label="Users (Synthetic)")
            n_days = gr.Slider(7, 60, value=30, step=1, label="Days (Synthetic)")
            ab_split = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Traffic Split to B (Synthetic)")
            conv_base = gr.Slider(0.01, 0.25, value=0.08, step=0.005, label="Baseline Conversion A (Synthetic)")
            conv_lift_b = gr.Slider(0.0, 0.1, value=0.02, step=0.005, label="Abs. Lift for B (Conversion)")
            ctr_base = gr.Slider(0.01, 0.6, value=0.18, step=0.01, label="Baseline CTR A")
            ctr_lift_b = gr.Slider(0.0, 0.2, value=0.03, step=0.005, label="Abs. Lift for B (CTR)")
            seed = gr.Number(value=42, label="Random Seed", precision=0)
            generate_btn = gr.Button("Load / Generate Data", variant="primary")

        with gr.Column(scale=1):
            data_info = gr.Markdown("### Data status: _no data yet_")
            users_df = gr.Dataframe(interactive=False, visible=False)
            sessions_df = gr.Dataframe(interactive=False, visible=False)

    state_users = gr.State()
    state_sessions = gr.State()
    state_retention = gr.State()
    state_days = gr.State()
    state_real = gr.State()  # bool flag for real-cookie mode

    def on_generate(ds, file, metric_label, n_users, n_days, ab_split, conv_base, conv_lift_b, ctr_base, ctr_lift_b, seed):
        if ds.startswith("Real"):
            if file is None:
                return "### Please upload the Cookie Cats CSV.", None, None, None, None, None, None, True
            users, err = load_cookie_cats_csv(file)
            if err:
                return f"### Error: {err}", None, None, None, None, None, None, True

            # Choose conversion metric
            if metric_label == "Retention (1-day)":
                users["signed_up"] = users["retention_1"].astype(int)
            else:
                users["signed_up"] = users["retention_7"].astype(int)

            # No sessions table / retention curve available in this dataset
            sessions = pd.DataFrame(columns=["user_id","variant","day","clicked","duration_s"])
            retention = pd.DataFrame(columns=["variant","day","active"])

            info = f"### Loaded Cookie Cats: **{len(users):,} users**. Variant split: A={ (users['variant']=='A').mean()*100:.1f}% / B={ (users['variant']=='B').mean()*100:.1f}%."
            return info, users.head(10), sessions.head(10), users, sessions, retention, 30, True

        # Synthetic path
        cfg = SimConfig(
            n_users=int(n_users), n_days=int(n_days), ab_split=float(ab_split),
            conv_base=float(conv_base), conv_lift_b=float(conv_lift_b),
            ctr_base=float(ctr_base), ctr_lift_b=float(ctr_lift_b), seed=int(seed)
        )
        users, sessions, retention = simulate_data(cfg)
        info = f"### Generated **{len(users):,} users** and **{len(sessions):,} sessions** over **{cfg.n_days} days**."
        return info, users.head(10), sessions.head(10), users, sessions, retention, int(cfg.n_days), False

    generate_btn.click(
        on_generate,
        inputs=[data_source, cookie_file, metric_choice, n_users, n_days, ab_split, conv_base, conv_lift_b, ctr_base, ctr_lift_b, seed],
        outputs=[data_info, users_df, sessions_df, state_users, state_sessions, state_retention, state_days, state_real]
    )

    gr.Markdown("---")

    with gr.Tabs():
        with gr.Tab("A/B Test + CUPED"):
            ab_table = gr.Dataframe(label="Variant Summary (%)", interactive=False)
            lift_md = gr.Markdown()
            cuped_md = gr.Markdown()

            def run_ab(users: pd.DataFrame):
                if users is None or len(users) == 0:
                    return None, "Load/generate data first.", "—"
                if "signed_up" not in users.columns:
                    return None, "Selected dataset lacks a conversion column.", "—"

                tbl = ab_summary(users)

                a = users.loc[users["variant"] == "A", "signed_up"]
                b = users.loc[users["variant"] == "B", "signed_up"]
                if len(a)==0 or len(b)==0:
                    return None, "Need both variants present.", "—"
                p1, n1 = float(a.mean()), int(len(a))
                p2, n2 = float(b.mean()), int(len(b))
                lift, (lo, hi), pval = lift_confint(p1, n1, p2, n2)
                lift_line = f"**Conversion (B - A)**: {lift*100:.2f} pp [{lo*100:.2f}, {hi*100:.2f}] • p-value = {pval:.4f}"

                # CUPED (note: in real data we're using a proxy covariate)
                y = users["signed_up"].astype(float).to_numpy()
                x = users["pre_exp_baseline"].astype(float).to_numpy()
                y_adj, vr_pct = cuped_adjustment(y, x)
                ua = users.copy(); ua["y_adj"] = y_adj
                a_adj = ua.loc[ua["variant"]=="A","y_adj"]; b_adj = ua.loc[ua["variant"]=="B","y_adj"]
                diff = float(b_adj.mean() - a_adj.mean())
                _, p_cuped = stats.ttest_ind(b_adj, a_adj, equal_var=False)
                cuped_line = f"**CUPED-adjusted difference (B - A)**: {diff*100:.2f} pp • p-value = {p_cuped:.4f}  \n_Variance reduction from CUPED: **{vr_pct:.1f}%**_"

                return tbl, lift_line, cuped_line

            gr.Button("Compute A/B Results").click(run_ab, inputs=state_users, outputs=[ab_table, lift_md, cuped_md])

        with gr.Tab("Retention & Funnel"):
            ret_plot = gr.Plot()
            fun_plot = gr.Plot()
            note_md = gr.Markdown()

            def run_retention(users: pd.DataFrame, retention_long: pd.DataFrame, real_cookie: bool):
                if users is None or len(users) == 0:
                    return None, None, "Load/generate data first."
                if real_cookie:
                    # Cookie Cats has retention_1/7 but not daily curves; show a small info note.
                    msg = "Cookie Cats dataset has `retention_1` and `retention_7` only (no daily session logs), so the retention curve and funnel are N/A."
                    return None, None, msg
                return retention_curve_plot(retention_long), funnel_plot(users), "OK"

            gr.Button("Compute Retention & Funnel").click(
                run_retention, inputs=[state_users, state_retention, state_real], outputs=[ret_plot, fun_plot, note_md]
            )

        with gr.Tab("Churn Model"):
            metrics_md = gr.Markdown()
            coef_tbl = gr.Dataframe(interactive=False)

            def run_churn(users: pd.DataFrame, sessions: pd.DataFrame, n_days: int, real_cookie: bool):
                if users is None or len(users) == 0:
                    return "Load/generate data first.", None
                m = churn_model(users, sessions, int(n_days) if n_days else 30, seed=19, real_cookie=bool(real_cookie))
                md = (
                    f"**Churn Model (Logistic Regression)**  \n"
                    f"AUC = **{m['auc']:.3f}**, Precision = **{m['precision']:.3f}**, "
                    f"Recall = **{m['recall']:.3f}**, F1 = **{m['f1']:.3f}**"
                )
                return md, m["coef_df"]

            gr.Button("Train & Evaluate").click(
                run_churn, inputs=[state_users, state_sessions, state_days, state_real], outputs=[metrics_md, coef_tbl]
            )

        with gr.Tab("Executive Summary"):
            summary_md = gr.Markdown()

            def build_summary(users: pd.DataFrame, real_cookie: bool):
                if users is None or len(users) == 0:
                    return "Load/generate data first."
                tbl = ab_summary(users).set_index("variant")
                if not {"A","B"}.issubset(tbl.index):
                    return "Need both variants."
                conv_a, conv_b = tbl.loc["A","conversion"], tbl.loc["B","conversion"]
                click_a, click_b = tbl.loc["A","clicked_any"], tbl.loc["B","clicked_any"]
                lift_conv = conv_b - conv_
                lift_click = click_b - click_a
                if real_cookie:
                    extra = "*(Real dataset: conversion = chosen retention metric)*"
                else:
                    extra = "*(Synthetic dataset)*"
                return (
                    f"### TL;DR\n"
                    f"- **Conversion**: A = **{conv_a:.2f}%**, B = **{conv_b:.2f}%** (lift **{lift_conv:+.2f} pp**)  {extra}\n"
                    f"- **CTR**: A = **{click_a:.2f}%**, B = **{click_b:.2f}%** (lift **{lift_click:+.2f} pp**)\n\n"
                    f"**Recommendation**: If significance holds (CUPED / t-test), scale Variant B while monitoring guardrails."
                )

            gr.Button("Generate Summary").click(build_summary, inputs=[state_users, state_real], outputs=summary_md)

    gr.Markdown(
        "---\n"
        "**Notes**: In real Cookie Cats mode, A/B is run on actual variants (`gate_30` vs `gate_40`) using `retention_1` or `retention_7` "
        "as the conversion metric. CUPED uses a proxy covariate derived from play volume; in production, prefer a true pre-experiment covariate "
        "or CUPAC. Retention curves/funnel aren’t available because the dataset lacks daily event logs."
    )

if __name__ == "__main__":
    demo.launch()
