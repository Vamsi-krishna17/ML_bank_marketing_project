import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib

matplotlib.use("Agg")

st.set_page_config(
    page_title="Bank Marketing ML Project",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .main { background-color: #f5f7fa; }
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #4f8ef7;
    }
    .card-red   { border-left: 5px solid #e74c3c; }
    .card-green { border-left: 5px solid #27ae60; }
    .card-gold  { border-left: 5px solid #f39c12; }
    .sec-title  { font-size: 26px; font-weight: 700; color: #1a1a2e; margin-bottom: 6px; }
    .sec-sub    { font-size: 14px; color: #666; margin-bottom: 20px; }
    .kpi-box {
        background: linear-gradient(135deg, #4f8ef7, #1a1a2e);
        color: white !important;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .kpi-box h2 { color: white !important; font-size: 32px; margin: 0; }
    .kpi-box p  { color: #cce0ff !important; font-size: 13px; margin: 0; }
    hr.fancy    { border: none; border-top: 2px solid #eee; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size": 12,
})

COLORS    = {"no": "#e74c3c", "yes": "#27ae60"}
BAR_COLOR = "#4f8ef7"


@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional-full.csv", sep=";")
    df.drop(columns=["default"], inplace=True)
    return df


def vlabels(ax, bars, fmt="{:,.0f}", fs=10):
    ymax = ax.get_ylim()[1]
    pad  = ymax * 0.012
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + pad,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fs, fontweight="bold")


def hlabels(ax, bars, fmt="{:,.0f}", fs=10):
    xmax = ax.get_xlim()[1]
    pad  = xmax * 0.012
    for b in bars:
        w = b.get_width()
        ax.text(w + pad, b.get_y() + b.get_height() / 2,
                fmt.format(w), ha="left", va="center",
                fontsize=fs, fontweight="bold")


with st.sidebar:
    st.markdown("## 🏦 Bank Marketing")
    st.markdown("### ML Project")
    st.markdown("---")

    pages = {
        "🏠  Home":                "Home",
        "📋  Business Problem":    "Business Problem",
        "🎯  Problem Statement":   "Problem Statement",
        "📦  Data Collection":     "Data Collection",
        "🔍  Dataset Overview":    "Dataset Overview",
        "📑  Feature Description": "Feature Description",
        "🧹  Data Cleaning":       "Data Cleaning",
        "📊  EDA":                 "EDA",
        "⚙️  Preprocessing":       "Preprocessing",
        "🤖  Model Building":      "Model Building",
        "🔮  Predict":             "Predict",
    }

    selection = st.radio("Navigate to", list(pages.keys()), label_visibility="collapsed")
    page = pages[selection]

    st.markdown("---")
    st.markdown("<small>Surisetti Vamsi Krishna<br>Data Science & ML Portfolio</small>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  HOME
# ════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown('<p class="sec-title">🏦 Bank Marketing Term Deposit Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">End-to-end Machine Learning project | Innomatics Research Labs</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>📌 About This Project</h4>
        <p>This project is built as part of the <b>Machine Learning Unique Project</b> at
        <b>Innomatics Research Labs</b>. It walks through the complete ML pipeline — business
        understanding, data collection, preprocessing, model building, evaluation, and
        deployment on <b>Streamlit</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="kpi-box"><h2>41,188</h2><p>Total Records</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-box"><h2>19</h2><p>Input Features</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-box"><h2>2</h2><p>Target Classes</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="kpi-box"><h2>Binary</h2><p>Classification Task</p></div>', unsafe_allow_html=True)

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Project Workflow")
    steps = [
        ("1️⃣", "Business Problem",    "Define the banking domain problem."),
        ("2️⃣", "Problem Statement",   "Formulate a precise ML objective."),
        ("3️⃣", "Data Collection",     "Source from UCI / Kaggle repository."),
        ("4️⃣", "Dataset Overview",    "Understand shape, types and stats."),
        ("5️⃣", "Feature Description", "Document every feature in detail."),
        ("6️⃣", "Data Cleaning",       "Handle nulls and dropped columns."),
        ("7️⃣", "EDA",                 "Univariate, Bivariate & Multivariate."),
        ("8️⃣", "Preprocessing",       "Encoding, Scaling & Train-Test Split."),
        ("9️⃣", "Modelling",           "Supervised Classification algorithms."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f'<div class="card"><b>{icon} {title}</b><br><small style="color:#666">{desc}</small></div>',
                        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  BUSINESS PROBLEM
# ════════════════════════════════════════════════════════════
elif page == "Business Problem":
    st.markdown('<p class="sec-title">📋 Business Problem</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Understanding the domain and the challenge faced by the bank</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>🏦 Domain: Banking & Finance</h4>
        <p>A Portuguese retail bank runs <b>telephone-based marketing campaigns</b> to
        sell <b>term deposit subscriptions</b>. Multiple calls are often made to the same
        client. The bank wants to use past data to identify which clients are most likely
        to subscribe — so resources are spent wisely.</p>
    </div>
    <div class="card card-red">
        <h4>❗ The Business Challenge</h4>
        <ul>
            <li>Cold-calling every client is <b>expensive and time-consuming</b>.</li>
            <li>Call-center agents have limited bandwidth — they need a <b>prioritised list</b>.</li>
            <li>Untargeted campaigns frustrate clients and <b>hurt the brand</b>.</li>
            <li>There is no current way to <b>predict subscription likelihood</b> before calling.</li>
        </ul>
    </div>
    <div class="card card-green">
        <h4>✅ Business Objective</h4>
        <p>Build a classification model that predicts whether a client will
        <b>subscribe to a term deposit</b> (yes / no) — enabling the bank to
        <b>target the right clients at the right time</b> and minimise missed subscribers.</p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.info("📉 **Cost Reduction**\n\nFewer unnecessary calls means lower operational cost.")
    with m2:
        st.success("📈 **Higher Conversion**\n\nFocus efforts only on likely subscribers.")
    with m3:
        st.warning("😊 **Better Experience**\n\nClients receive relevant calls only.")


# ════════════════════════════════════════════════════════════
#  PROBLEM STATEMENT
# ════════════════════════════════════════════════════════════
elif page == "Problem Statement":
    st.markdown('<p class="sec-title">🎯 Problem Statement</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">ML formulation of the business challenge</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>📝 Problem Statement</h4>
        <p style="font-size:16px; line-height:1.9">
        Given historical data of bank clients and their interactions during past telephone
        marketing campaigns, build a <b>supervised binary classification model</b> that
        predicts whether a client will <b>subscribe to a term deposit</b>
        — target variable <code>y</code> : <b>yes</b> or <b>no</b>.
        </p>
    </div>
    <div class="card card-green">
        <h4>🤖 Supervised Learning — Classification Task</h4>
        <ul>
            <li><b>Target variable:</b> <code>y</code> (yes = 1, no = 0)</li>
            <li><b>Algorithms used:</b> Logistic Regression, Random Forest, XGBoost,
                Gradient Boosting, AdaBoost</li>
            <li><b>Balancing:</b> SMOTE and ADASYN applied to handle class imbalance</li>
            <li><b>Primary metric:</b> Recall — minimize missed subscribers (False Negatives)</li>
            <li><b>Secondary metrics:</b> ROC-AUC, F1-Score, Precision, Accuracy</li>
        </ul>
    </div>
    <div class="card card-gold">
        <h4>📌 Key Assumptions</h4>
        <ul>
            <li>Data is representative of the real client population.</li>
            <li>Phone calls are the sole marketing channel in this dataset.</li>
            <li>Economic indicators are external and not directly controllable.</li>
            <li>Class imbalance (majority = "no") is handled using SMOTE and ADASYN.</li>
            <li><code>default</code> column was dropped — too many unknowns, near-zero variance.</li>
            <li>Minimising <b>False Negatives</b> (missed subscribers) is the core business priority.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown('<div class="card"><b>ROC-AUC</b><br><small>Not affected by class imbalance. Measures overall separability.</small></div>', unsafe_allow_html=True)
    with e2:
        st.markdown('<div class="card card-green"><b>Recall (Primary)</b><br><small>Missing a subscriber = direct revenue loss. Recall must be maximised.</small></div>', unsafe_allow_html=True)
    with e3:
        st.markdown('<div class="card card-gold"><b>F1-Score</b><br><small>Balances precision and recall — secondary evaluation metric.</small></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  DATA COLLECTION
# ════════════════════════════════════════════════════════════
elif page == "Data Collection":
    st.markdown('<p class="sec-title">📦 Data Collection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Source, background, and acquisition method of the dataset</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h4>📂 Dataset Details</h4>
        <table width="100%">
            <tr><td><b>Name</b></td><td>Bank Marketing Dataset (Full)</td></tr>
            <tr><td><b>File</b></td><td><code>bank-additional-full.csv</code></td></tr>
            <tr><td><b>Source</b></td><td>UCI Machine Learning Repository</td></tr>
            <tr><td><b>Delimiter</b></td><td>Semicolon ( <code>;</code> )</td></tr>
            <tr><td><b>Rows</b></td><td>41,188</td></tr>
            <tr><td><b>Columns</b></td><td>21 original → 20 after dropping <code>default</code></td></tr>
            <tr><td><b>Access</b></td><td>Open / Public — no authentication required</td></tr>
        </table>
    </div>
    <div class="card card-green">
        <h4>🔗 How It Was Collected</h4>
        <p>Downloaded directly from the <b>UCI ML Repository</b> and loaded using:</p>
        <pre><code>df = pd.read_csv("bank-additional-full.csv", sep=";")</code></pre>
        <p>No web scraping, APIs, or authentication were needed.</p>
    </div>
    <div class="card card-red">
        <h4>📖 Dataset Background</h4>
        <p>Related to <b>direct marketing campaigns</b> of a Portuguese banking institution
        conducted via <b>phone calls</b>.<br>
        <b>Reference:</b> S. Moro, P. Cortez and P. Rita. (2014).
        <i>A Data-Driven Approach to Predict the Success of Bank Telemarketing.</i>
        Decision Support Systems, Elsevier.</p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  DATASET OVERVIEW
# ════════════════════════════════════════════════════════════
elif page == "Dataset Overview":
    st.markdown('<p class="sec-title">🔍 Dataset Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Shape, data types, missing values and basic statistics</p>', unsafe_allow_html=True)

    try:
        df = load_data()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-box"><h2>{df.shape[0]:,}</h2><p>Total Rows</p></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-box"><h2>{df.shape[1]}</h2><p>Total Columns</p></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-box"><h2>{df.select_dtypes(include=np.number).shape[1]}</h2><p>Numerical</p></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="kpi-box"><h2>{df.select_dtypes(include="object").shape[1]}</h2><p>Categorical</p></div>', unsafe_allow_html=True)

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
        st.markdown("#### 👁️ Raw Data Preview")
        n = st.slider("Rows to display", 5, 50, 10)
        st.dataframe(df.head(n), use_container_width=True)

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🧬 Column Info")
            info_df = pd.DataFrame({
                "Column":        df.columns,
                "Dtype":         df.dtypes.astype(str).values,
                "Non-Null":      df.notnull().sum().values,
                "Null":          df.isnull().sum().values,
                "Unique Values": df.nunique().values,
            })
            st.dataframe(info_df, use_container_width=True, height=560)

        with col_b:
            st.markdown("#### ❓ Missing Values")
            missing = df.isnull().sum()
            if missing.sum() == 0:
                st.success("✅ No missing values found in any column.")
            else:
                st.dataframe(pd.DataFrame({
                    "Column":    missing[missing > 0].index,
                    "Missing":   missing[missing > 0].values,
                    "% Missing": (missing[missing > 0] / len(df) * 100).round(2).values,
                }), use_container_width=True)

            st.markdown("#### 🎯 Target Variable — `y`")
            vc = df["y"].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            vc["Percentage (%)"] = (vc["Count"] / len(df) * 100).round(2)
            st.dataframe(vc, use_container_width=True)
            st.warning("⚠️ **Class Imbalance** — 'no' dominates. SMOTE and ADASYN used to balance.")

    except FileNotFoundError:
        st.error("❌ Place `bank-additional-full.csv` in the same folder as app.py.")


# ════════════════════════════════════════════════════════════
#  FEATURE DESCRIPTION
# ════════════════════════════════════════════════════════════
elif page == "Feature Description":
    st.markdown('<p class="sec-title">📑 Feature Description</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">All 20 features documented — <code>default</code> column excluded</p>', unsafe_allow_html=True)

    feature_data = [
        ("age",            "Numerical",   "Client",     "Age of the client in years.",                                                      "Integer"),
        ("job",            "Categorical", "Client",     "Type of job held by the client.",                                                  "admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown"),
        ("marital",        "Categorical", "Client",     "Marital status.",                                                                  "divorced, married, single, unknown"),
        ("education",      "Categorical", "Client",     "Highest education level attained.",                                                "basic.4y, basic.6y, basic.9y, high.school, illiterate, professional.course, university.degree, unknown"),
        ("housing",        "Binary",      "Client",     "Does the client have a housing loan?",                                             "yes, no, unknown"),
        ("loan",           "Binary",      "Client",     "Does the client have a personal loan?",                                            "yes, no, unknown"),
        ("contact",        "Categorical", "Campaign",   "Type of communication used for contact.",                                          "cellular, telephone"),
        ("month",          "Categorical", "Campaign",   "Last contact month of the year.",                                                  "jan to dec"),
        ("day_of_week",    "Categorical", "Campaign",   "Last contact day of the week.",                                                    "mon, tue, wed, thu, fri"),
        ("duration",       "Numerical",   "Campaign",   "Last call duration in seconds. Not available before call — benchmark only.",       "Integer (seconds)"),
        ("campaign",       "Numerical",   "Campaign",   "Number of contacts during this campaign for this client.",                        "Integer"),
        ("pdays",          "Numerical",   "Previous",   "Days since last contacted in previous campaign. 999 = not previously contacted.", "Integer"),
        ("previous",       "Numerical",   "Previous",   "Number of contacts before this campaign.",                                        "Integer"),
        ("poutcome",       "Categorical", "Previous",   "Outcome of the previous marketing campaign.",                                     "failure, nonexistent, success"),
        ("emp.var.rate",   "Numerical",   "Social-Eco", "Employment variation rate — quarterly indicator.",                                "Float"),
        ("cons.price.idx", "Numerical",   "Social-Eco", "Consumer price index — monthly indicator.",                                       "Float"),
        ("cons.conf.idx",  "Numerical",   "Social-Eco", "Consumer confidence index — monthly indicator.",                                  "Float"),
        ("euribor3m",      "Numerical",   "Social-Eco", "Euribor 3-month rate — daily indicator.",                                         "Float"),
        ("nr.employed",    "Numerical",   "Social-Eco", "Number of employees quarterly in thousands.",                                     "Float"),
        ("y",              "Target",      "Output",     "Has the client subscribed to a term deposit?",                                    "yes / no"),
    ]

    feat_df = pd.DataFrame(feature_data,
                           columns=["Feature", "Type", "Category", "Description", "Values / Range"])

    fc1, fc2 = st.columns(2)
    with fc1:
        type_filter = st.multiselect("Filter by Type",
            ["Numerical", "Categorical", "Binary", "Target"],
            default=["Numerical", "Categorical", "Binary", "Target"])
    with fc2:
        cat_filter = st.multiselect("Filter by Category",
            ["Client", "Campaign", "Previous", "Social-Eco", "Output"],
            default=["Client", "Campaign", "Previous", "Social-Eco", "Output"])

    filtered = feat_df[feat_df["Type"].isin(type_filter) & feat_df["Category"].isin(cat_filter)]

    def style_type(col):
        m = {
            "Numerical":   "background-color:#dff0d8; color:#3c763d",
            "Categorical": "background-color:#d9edf7; color:#31708f",
            "Binary":      "background-color:#fcf8e3; color:#8a6d3b",
            "Target":      "background-color:#f2dede; color:#a94442",
        }
        return [m.get(v, "") for v in col]

    st.dataframe(filtered.style.apply(style_type, subset=["Type"]),
                 use_container_width=True, height=680)

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
    for cat, icon in [("Client","👤"),("Campaign","📞"),("Previous","🔄"),("Social-Eco","🌍"),("Output","🎯")]:
        grp   = feat_df[feat_df["Category"] == cat]
        names = "  ·  ".join(f"`{f}`" for f in grp["Feature"].tolist())
        st.markdown(f"**{icon} {cat}** ({len(grp)}):  {names}")

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
    counts = feat_df["Type"].value_counts()
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="card card-green"><b>🔢 Numerical</b><br><h2>{counts.get("Numerical",0)}</h2></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="card"><b>🔤 Categorical</b><br><h2>{counts.get("Categorical",0)}</h2></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="card card-gold"><b>⚡ Binary</b><br><h2>{counts.get("Binary",0)}</h2></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="card card-red"><b>🎯 Target</b><br><h2>{counts.get("Target",0)}</h2></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  DATA CLEANING
# ════════════════════════════════════════════════════════════
elif page == "Data Cleaning":
    st.markdown('<p class="sec-title">🧹 Data Cleaning</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Missing values, unknown categories, outlier strategy, and column removal</p>', unsafe_allow_html=True)

    try:
        df = load_data()
        dc1, dc2 = st.tabs(["🔍 Checks & Findings", "🗑️ Column Removal"])

        with dc1:
            st.markdown("#### ✅ Missing Values")
            if df.isnull().sum().sum() == 0:
                st.success("No null values found in any column after loading.")
            else:
                st.dataframe(df.isnull().sum()[df.isnull().sum() > 0], use_container_width=True)

            st.markdown("#### ❓ 'unknown' Entries per Column")
            unk = (df == "unknown").sum()
            unk = unk[unk > 0].reset_index()
            unk.columns = ["Column", "Unknown Count"]
            unk["% of Total"] = (unk["Unknown Count"] / len(df) * 100).round(2)
            st.dataframe(unk, use_container_width=True)
            st.info("'unknown' values are retained as a valid category — they may carry predictive signal.")

            st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
            st.markdown("""
            <div class="card card-gold">
                <h4>📌 Outlier Handling Strategy</h4>
                <p>No rows were removed or clipped. Every record represents a real client call.
                <b>Robust Scaler</b> handles outliers during preprocessing — it scales
                based on IQR (Q1–Q3) and is insensitive to extreme values.</p>
            </div>
            """, unsafe_allow_html=True)

        with dc2:
            st.markdown("""
            <div class="card card-red">
                <h4>🗑️ Dropped Column: <code>default</code></h4>
                <ul>
                    <li>~20% of values are "unknown" — too many to be reliable.</li>
                    <li>Near-zero variance — almost all values are "no".</li>
                    <li>Adds noise rather than predictive signal.</li>
                </ul>
                <pre><code>df.drop(columns=["default"], inplace=True)</code></pre>
            </div>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Columns Before", 21)
            c2.metric("Columns After",  df.shape[1])
            c3.metric("Dropped", 1, delta=-1)
            st.markdown("#### 📋 Cleaned Dataset — First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)

    except FileNotFoundError:
        st.error("❌ `bank-additional-full.csv` not found.")


# ════════════════════════════════════════════════════════════
#  EDA
# ════════════════════════════════════════════════════════════
elif page == "EDA":
    st.markdown('<p class="sec-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Select a column and plot type to explore the data visually</p>', unsafe_allow_html=True)

    try:
        df             = load_data()
        num_cols       = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols       = df.select_dtypes(include="object").columns.tolist()
        cat_cols_notgt = [c for c in cat_cols if c != "y"]

        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📈 Univariate", "🔗 Bivariate", "🌐 Multivariate"])

        with eda_tab1:
            u_num_tab, u_cat_tab = st.tabs(["🔢 Numerical", "🔤 Categorical"])

            with u_num_tab:
                col1, col2 = st.columns(2)
                with col1:
                    sel_num = st.selectbox("Select numerical column", num_cols, key="u_num")
                with col2:
                    num_plot = st.selectbox("Plot type",
                        ["Histogram", "Box Plot", "Histogram + Box Plot", "KDE", "Violin Plot"],
                        key="u_num_plt")

                s = df[sel_num]

                if num_plot == "Histogram":
                    fig, ax = plt.subplots(figsize=(14, 6))
                    cnts, _, patches = ax.hist(s, bins=35, color=BAR_COLOR, edgecolor="white", alpha=0.85)
                    for i, (p, c) in enumerate(zip(patches, cnts)):
                        if c > 0 and i % 3 == 0:
                            ax.text(p.get_x() + p.get_width()/2, p.get_height() + max(cnts)*0.005,
                                    f"{int(c):,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
                    ax.set_title(f"Distribution of {sel_num}", fontsize=15, fontweight="bold", pad=12)
                    ax.set_xlabel(sel_num, fontsize=13)
                    ax.set_ylabel("Frequency", fontsize=13)
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif num_plot == "Box Plot":
                    fig, ax = plt.subplots(figsize=(8, 7))
                    ax.boxplot(s.dropna(), patch_artist=True, widths=0.5,
                               boxprops=dict(facecolor="#d9edf7", color="#31708f", linewidth=1.5),
                               medianprops=dict(color="#e74c3c", linewidth=2.5),
                               whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
                               flierprops=dict(marker="o", markersize=3, alpha=0.4))
                    ax.set_title(f"Box Plot — {sel_num}", fontsize=15, fontweight="bold", pad=12)
                    ax.set_ylabel(sel_num, fontsize=13)
                    ax.set_xticks([])
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif num_plot == "Histogram + Box Plot":
                    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                    cnts, _, patches = axes[0].hist(s, bins=35, color=BAR_COLOR, edgecolor="white", alpha=0.85)
                    for i, (p, c) in enumerate(zip(patches, cnts)):
                        if c > 0 and i % 3 == 0:
                            axes[0].text(p.get_x() + p.get_width()/2, p.get_height() + max(cnts)*0.005,
                                         f"{int(c):,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
                    axes[0].set_title(f"Histogram — {sel_num}", fontsize=14, fontweight="bold")
                    axes[0].set_xlabel(sel_num, fontsize=12)
                    axes[0].set_ylabel("Frequency", fontsize=12)
                    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
                    axes[1].boxplot(s.dropna(), patch_artist=True, widths=0.5,
                                    boxprops=dict(facecolor="#d9edf7", color="#31708f", linewidth=1.5),
                                    medianprops=dict(color="#e74c3c", linewidth=2.5),
                                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
                    axes[1].set_title(f"Box Plot — {sel_num}", fontsize=14, fontweight="bold")
                    axes[1].set_ylabel(sel_num, fontsize=12)
                    axes[1].set_xticks([])
                    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif num_plot == "KDE":
                    fig, ax = plt.subplots(figsize=(14, 6))
                    s.plot.kde(ax=ax, color=BAR_COLOR, linewidth=2.5)
                    line = ax.get_lines()[0]
                    ax.fill_between(line.get_xdata(), line.get_ydata(), alpha=0.2, color=BAR_COLOR)
                    ax.set_title(f"KDE Density — {sel_num}", fontsize=15, fontweight="bold", pad=12)
                    ax.set_xlabel(sel_num, fontsize=13)
                    ax.set_ylabel("Density", fontsize=13)
                    ax.grid(linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif num_plot == "Violin Plot":
                    fig, ax = plt.subplots(figsize=(8, 7))
                    parts = ax.violinplot(s.dropna(), showmedians=True, showextrema=True)
                    parts["cmedians"].set_color("#e74c3c")
                    parts["cmedians"].set_linewidth(2.5)
                    for pc in parts["bodies"]:
                        pc.set_facecolor(BAR_COLOR)
                        pc.set_alpha(0.6)
                    ax.set_title(f"Violin Plot — {sel_num}", fontsize=15, fontweight="bold", pad=12)
                    ax.set_ylabel(sel_num, fontsize=13)
                    ax.set_xticks([])
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                m1c, m2c, m3c, m4c, m5c = st.columns(5)
                m1c.metric("Mean",     f"{s.mean():.2f}")
                m2c.metric("Median",   f"{s.median():.2f}")
                m3c.metric("Std Dev",  f"{s.std():.2f}")
                m4c.metric("Skewness", f"{s.skew():.2f}")
                m5c.metric("Kurtosis", f"{s.kurt():.2f}")

                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR    = Q3 - Q1
                outs   = s[(s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)]
                st.markdown(f"""
                <div class="card card-gold">
                    <b>🔍 IQR Outlier Check</b>&nbsp;
                    Q1=<b>{Q1:.2f}</b> | Q3=<b>{Q3:.2f}</b> | IQR=<b>{IQR:.2f}</b> |
                    Lower=<b>{Q1-1.5*IQR:.2f}</b> | Upper=<b>{Q3+1.5*IQR:.2f}</b><br>
                    Outliers: <b>{len(outs)}</b> ({len(outs)/len(s)*100:.2f}% of rows)
                </div>
                """, unsafe_allow_html=True)

            with u_cat_tab:
                col1, col2 = st.columns(2)
                with col1:
                    sel_cat = st.selectbox("Select categorical column", cat_cols, key="u_cat")
                with col2:
                    cat_plot = st.selectbox("Plot type",
                        ["Vertical Bar", "Horizontal Bar", "Pie Chart"], key="u_cat_plt")

                vc = df[sel_cat].value_counts()

                if cat_plot == "Vertical Bar":
                    fig, ax = plt.subplots(figsize=(max(12, len(vc)*1.6), 7))
                    bars = ax.bar(range(len(vc)), vc.values, color=BAR_COLOR, edgecolor="white", width=0.6)
                    ax.set_xticks(range(len(vc)))
                    ax.set_xticklabels(vc.index, rotation=35, ha="right", fontsize=11)
                    vlabels(ax, bars)
                    ax.set_title(f"Value Counts — {sel_cat}", fontsize=15, fontweight="bold", pad=12)
                    ax.set_xlabel(sel_cat, fontsize=13)
                    ax.set_ylabel("Count", fontsize=13)
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif cat_plot == "Horizontal Bar":
                    fig, ax = plt.subplots(figsize=(13, max(6, len(vc)*0.75)))
                    bars = ax.barh(range(len(vc)), vc.values, color=BAR_COLOR, edgecolor="white", height=0.6)
                    ax.set_yticks(range(len(vc)))
                    ax.set_yticklabels(vc.index, fontsize=11)
                    hlabels(ax, bars)
                    ax.set_title(f"Value Counts — {sel_cat}", fontsize=15, fontweight="bold", pad=12)
                    ax.set_ylabel(sel_cat, fontsize=13)
                    ax.set_xlabel("Count", fontsize=13)
                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                    ax.grid(axis="x", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif cat_plot == "Pie Chart":
                    fig, ax = plt.subplots(figsize=(9, 7))
                    wedges, texts, autotexts = ax.pie(
                        vc.values, labels=vc.index,
                        autopct="%1.1f%%", startangle=140,
                        colors=plt.cm.Set3.colors, pctdistance=0.80)
                    for t in texts:
                        t.set_fontsize(11)
                    for t in autotexts:
                        t.set_fontsize(10)
                        t.set_fontweight("bold")
                    ax.set_title(f"Distribution — {sel_cat}", fontsize=15, fontweight="bold", pad=12)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                st.dataframe(pd.DataFrame({
                    "Category":       vc.index,
                    "Count":          vc.values,
                    "Percentage (%)": (vc.values / len(df) * 100).round(2),
                }), use_container_width=True)

        with eda_tab2:
            b_cat_tab, b_num_tab, b_nn_tab = st.tabs([
                "📊 Categorical vs Target",
                "🔢 Numerical vs Target",
                "🔢 Num vs Num",
            ])

            with b_cat_tab:
                col1, col2 = st.columns(2)
                with col1:
                    sel_cat_bi = st.selectbox("Select categorical feature", cat_cols_notgt, key="bi_cat")
                with col2:
                    cat_bi_plt = st.selectbox("Plot type",
                        ["Grouped Bar (%)", "Stacked % Bar", "Count Bar"], key="bi_cat_plt")

                ct_pct = pd.crosstab(df[sel_cat_bi], df["y"], normalize="index") * 100
                ct_cnt = pd.crosstab(df[sel_cat_bi], df["y"])
                cats_v = ct_pct.index.tolist()
                x      = np.arange(len(cats_v))

                if cat_bi_plt == "Grouped Bar (%)":
                    fig, ax = plt.subplots(figsize=(max(13, len(cats_v)*1.5), 7))
                    w = 0.35
                    b_no  = ax.bar(x-w/2, ct_pct["no"],  width=w, label="No",  color=COLORS["no"],  edgecolor="white")
                    b_yes = ax.bar(x+w/2, ct_pct["yes"], width=w, label="Yes", color=COLORS["yes"], edgecolor="white")
                    vlabels(ax, b_no,  fmt="{:.1f}%", fs=9)
                    vlabels(ax, b_yes, fmt="{:.1f}%", fs=9)
                    ax.set_xticks(x)
                    ax.set_xticklabels(cats_v, rotation=35, ha="right", fontsize=11)
                    ax.set_title(f"Subscription Rate (%) by {sel_cat_bi}", fontsize=15, fontweight="bold", pad=12)
                    ax.set_xlabel(sel_cat_bi, fontsize=13)
                    ax.set_ylabel("Percentage (%)", fontsize=13)
                    ax.legend(title="Subscribed", fontsize=11)
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif cat_bi_plt == "Stacked % Bar":
                    fig, ax = plt.subplots(figsize=(max(13, len(cats_v)*1.5), 7))
                    ax.bar(x, ct_pct["no"],  width=0.55, label="No",  color=COLORS["no"],  edgecolor="white")
                    ax.bar(x, ct_pct["yes"], width=0.55, label="Yes", color=COLORS["yes"], edgecolor="white", bottom=ct_pct["no"].values)
                    for i, (pn, py) in enumerate(zip(ct_pct["no"], ct_pct["yes"])):
                        if pn > 5:
                            ax.text(i, pn/2,     f"{pn:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
                        if py > 5:
                            ax.text(i, pn+py/2, f"{py:.1f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
                    ax.set_xticks(x)
                    ax.set_xticklabels(cats_v, rotation=35, ha="right", fontsize=11)
                    ax.set_title(f"Stacked Subscription Rate (%) by {sel_cat_bi}", fontsize=15, fontweight="bold", pad=12)
                    ax.set_xlabel(sel_cat_bi, fontsize=13)
                    ax.set_ylabel("Percentage (%)", fontsize=13)
                    ax.legend(title="Subscribed", fontsize=11)
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif cat_bi_plt == "Count Bar":
                    fig, ax = plt.subplots(figsize=(max(13, len(cats_v)*1.5), 7))
                    w = 0.35
                    b_no  = ax.bar(x-w/2, ct_cnt["no"],  width=w, label="No",  color=COLORS["no"],  edgecolor="white")
                    b_yes = ax.bar(x+w/2, ct_cnt["yes"], width=w, label="Yes", color=COLORS["yes"], edgecolor="white")
                    vlabels(ax, b_no,  fs=9)
                    vlabels(ax, b_yes, fs=9)
                    ax.set_xticks(x)
                    ax.set_xticklabels(cats_v, rotation=35, ha="right", fontsize=11)
                    ax.set_title(f"Subscription Count by {sel_cat_bi}", fontsize=15, fontweight="bold", pad=12)
                    ax.set_xlabel(sel_cat_bi, fontsize=13)
                    ax.set_ylabel("Count", fontsize=13)
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                    ax.legend(title="Subscribed", fontsize=11)
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                st.dataframe(ct_pct.round(2).rename(columns={"no":"No (%)","yes":"Yes (%)"}),
                             use_container_width=True)

            with b_num_tab:
                col1, col2 = st.columns(2)
                with col1:
                    sel_num_bi = st.selectbox("Select numerical feature", num_cols, key="bi_num")
                with col2:
                    num_bi_plt = st.selectbox("Plot type",
                        ["Overlapping Histogram","KDE Comparison","Box Plot by Target","Violin by Target"],
                        key="bi_num_plt")

                if num_bi_plt == "Overlapping Histogram":
                    fig, ax = plt.subplots(figsize=(14, 7))
                    for lbl, col in COLORS.items():
                        ax.hist(df[df["y"]==lbl][sel_num_bi], bins=35, alpha=0.65, color=col, label=lbl, edgecolor="white")
                    ax.set_title(f"{sel_num_bi} Distribution by Target", fontsize=15, fontweight="bold", pad=12)
                    ax.set_xlabel(sel_num_bi, fontsize=13)
                    ax.set_ylabel("Frequency", fontsize=13)
                    ax.legend(title="Subscribed (y)", fontsize=11)
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif num_bi_plt == "KDE Comparison":
                    fig, ax = plt.subplots(figsize=(14, 7))
                    for lbl, col in COLORS.items():
                        df[df["y"]==lbl][sel_num_bi].plot.kde(ax=ax, label=lbl, color=col, linewidth=2.5)
                    ax.set_title(f"KDE — {sel_num_bi} by Target", fontsize=15, fontweight="bold", pad=12)
                    ax.set_xlabel(sel_num_bi, fontsize=13)
                    ax.set_ylabel("Density", fontsize=13)
                    ax.legend(title="Subscribed (y)", fontsize=11)
                    ax.grid(linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif num_bi_plt == "Box Plot by Target":
                    groups = [df[df["y"]==lbl][sel_num_bi].dropna() for lbl in ["no","yes"]]
                    fig, ax = plt.subplots(figsize=(9, 7))
                    bp = ax.boxplot(groups, patch_artist=True, labels=["No","Yes"], widths=0.45,
                                    medianprops=dict(color="#333", linewidth=2.5),
                                    flierprops=dict(marker="o", markersize=3, alpha=0.35))
                    bp["boxes"][0].set_facecolor(COLORS["no"]  + "99")
                    bp["boxes"][1].set_facecolor(COLORS["yes"] + "99")
                    ax.set_title(f"{sel_num_bi} by Target", fontsize=15, fontweight="bold", pad=12)
                    ax.set_ylabel(sel_num_bi, fontsize=13)
                    ax.set_xlabel("Subscribed (y)", fontsize=13)
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif num_bi_plt == "Violin by Target":
                    groups = [df[df["y"]==lbl][sel_num_bi].dropna() for lbl in ["no","yes"]]
                    fig, ax = plt.subplots(figsize=(9, 7))
                    parts = ax.violinplot(groups, showmedians=True)
                    for pc, col in zip(parts["bodies"], [COLORS["no"], COLORS["yes"]]):
                        pc.set_facecolor(col)
                        pc.set_alpha(0.65)
                    ax.set_xticks([1, 2])
                    ax.set_xticklabels(["No","Yes"], fontsize=12)
                    ax.set_title(f"{sel_num_bi} by Target", fontsize=15, fontweight="bold", pad=12)
                    ax.set_ylabel(sel_num_bi, fontsize=13)
                    ax.set_xlabel("Subscribed (y)", fontsize=13)
                    ax.grid(axis="y", linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                st.dataframe(df.groupby("y")[sel_num_bi].describe().round(2), use_container_width=True)

            with b_nn_tab:
                col1, col2, col3 = st.columns(3)
                with col1:
                    col_x = st.selectbox("X-axis", num_cols, key="nn_x")
                with col2:
                    col_y = st.selectbox("Y-axis", [c for c in num_cols if c != col_x], key="nn_y")
                with col3:
                    nn_plt = st.selectbox("Plot type",
                        ["Scatter (colored by y)","Hexbin","Line (sorted)"], key="nn_plt")

                if nn_plt == "Scatter (colored by y)":
                    fig, ax = plt.subplots(figsize=(14, 7))
                    for lbl, col in COLORS.items():
                        sub = df[df["y"]==lbl]
                        ax.scatter(sub[col_x], sub[col_y], alpha=0.3, s=10, color=col, label=lbl)
                    ax.set_xlabel(col_x, fontsize=13)
                    ax.set_ylabel(col_y, fontsize=13)
                    ax.set_title(f"{col_x} vs {col_y} — colored by target", fontsize=15, fontweight="bold", pad=12)
                    ax.legend(title="Subscribed (y)", fontsize=11)
                    ax.grid(linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif nn_plt == "Hexbin":
                    fig, ax = plt.subplots(figsize=(14, 7))
                    hb = ax.hexbin(df[col_x], df[col_y], gridsize=35, cmap="Blues", mincnt=1)
                    plt.colorbar(hb, ax=ax, label="Count")
                    ax.set_xlabel(col_x, fontsize=13)
                    ax.set_ylabel(col_y, fontsize=13)
                    ax.set_title(f"{col_x} vs {col_y} — Hexbin Density", fontsize=15, fontweight="bold", pad=12)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                elif nn_plt == "Line (sorted)":
                    samp = df[[col_x, col_y]].sample(800, random_state=42).sort_values(col_x)
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(samp[col_x], samp[col_y], color=BAR_COLOR, linewidth=0.9, alpha=0.75)
                    ax.set_xlabel(col_x, fontsize=13)
                    ax.set_ylabel(col_y, fontsize=13)
                    ax.set_title(f"{col_x} vs {col_y} — Line (sample 800)", fontsize=15, fontweight="bold", pad=12)
                    ax.grid(linestyle="--", alpha=0.4)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                corr_val = df[[col_x, col_y]].corr().iloc[0, 1]
                st.info(f"📐 Pearson Correlation — **{col_x}** vs **{col_y}** : **{corr_val:.4f}**")

        with eda_tab3:
            m_heat, m_pair = st.tabs(["🌡️ Correlation Heatmap", "📉 Pair Plot"])

            with m_heat:
                hm_cols  = st.multiselect("Select numerical columns", num_cols, default=num_cols, key="hm_cols")
                hm_annot = st.checkbox("Show values on heatmap", value=True)

                if len(hm_cols) >= 2:
                    st.markdown("""
                    <div class="card card-gold">
                        <b>📌 Multicollinearity Note</b> — <code>emp.var.rate</code>,
                        <code>euribor3m</code>, and <code>nr.employed</code> are strongly
                        correlated macro-economic indicators. Handled via Robust Scaling.
                    </div>
                    """, unsafe_allow_html=True)
                    corr = df[hm_cols].corr()
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    fig, ax = plt.subplots(figsize=(max(11, len(hm_cols)*1.1), max(8, len(hm_cols)*0.9)))
                    sns.heatmap(corr, annot=hm_annot, fmt=".2f", cmap="coolwarm",
                                linewidths=0.5, ax=ax, mask=mask,
                                annot_kws={"size":10}, vmin=-1, vmax=1)
                    ax.set_title("Correlation Heatmap (lower triangle)", fontsize=14, fontweight="bold", pad=12)
                    plt.xticks(rotation=35, ha="right", fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                else:
                    st.warning("Select at least 2 columns.")

            with m_pair:
                pair_feats  = st.multiselect("Select features (3–5 recommended)",
                    options=num_cols, default=["age","duration","campaign","euribor3m"], key="pair_cols")
                pair_sample = st.slider("Sample size", 500, 3000, 1500, step=500)

                if len(pair_feats) >= 2:
                    pdf = df[pair_feats+["y"]].sample(min(pair_sample, len(df)), random_state=42)
                    n   = len(pair_feats)
                    fig = plt.figure(figsize=(max(10, n*2.8), max(8, n*2.8)))
                    for i, fx in enumerate(pair_feats):
                        for j, fy in enumerate(pair_feats):
                            ax = fig.add_subplot(n, n, i*n+j+1)
                            if i == j:
                                for lbl, col in COLORS.items():
                                    ax.hist(pdf[pdf["y"]==lbl][fx], bins=20, alpha=0.6, color=col, density=True)
                            else:
                                for lbl, col in COLORS.items():
                                    sub = pdf[pdf["y"]==lbl]
                                    ax.scatter(sub[fy], sub[fx], alpha=0.2, s=5, color=col)
                            if i == n-1: ax.set_xlabel(fy, fontsize=8)
                            if j == 0:   ax.set_ylabel(fx, fontsize=8)
                            ax.tick_params(labelsize=7)
                    plt.suptitle(f"Pair Plot (n={pair_sample})", fontweight="bold", fontsize=13, y=1.01)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                else:
                    st.info("Select at least 2 features.")

    except FileNotFoundError:
        st.error("❌ `bank-additional-full.csv` not found.")


# ════════════════════════════════════════════════════════════
#  PREPROCESSING
# ════════════════════════════════════════════════════════════
elif page == "Preprocessing":
    st.markdown('<p class="sec-title">⚙️ Data Preprocessing</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Train-Test Split → One-Hot Encoding → Robust Scaling → Label Encoding</p>', unsafe_allow_html=True)

    try:
        from sklearn.preprocessing import RobustScaler, LabelEncoder
        from sklearn.model_selection import train_test_split

        df = load_data()
        X  = df.drop(columns=["y"])
        Y  = df["y"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, train_size=0.8, test_size=0.2, random_state=42)

        categorical_cols = [
            "job","marital","education","housing","loan",
            "contact","month","day_of_week","poutcome",
        ]

        X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, dtype=int)
        X_test_encoded  = pd.get_dummies(X_test,  columns=categorical_cols, dtype=int)
        X_test_encoded  = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

        rs         = RobustScaler()
        X_train_sc = rs.fit_transform(X_train_encoded)
        X_test_sc  = rs.transform(X_test_encoded)

        le          = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)

        def vlabels_pp(ax, bars, fs=11):
            ymax = ax.get_ylim()[1]
            pad  = ymax * 0.012
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h+pad,
                        f"{int(h):,}", ha="center", va="bottom",
                        fontsize=fs, fontweight="bold")

        pp1, pp2, pp3, pp4, pp5 = st.tabs([
            "✂️ Train-Test Split", "🔡 One-Hot Encoding",
            "📏 Robust Scaling",   "🏷️ Label Encoding", "📋 Final Summary",
        ])

        with pp1:
            st.markdown("""
            <div class="card">
                <h4>✂️ Train-Test Split — 80 / 20</h4>
                <pre><code>X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=42)</code></pre>
            </div>
            """, unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("X_train", f"{X_train.shape[0]:,} rows")
            s2.metric("X_test",  f"{X_test.shape[0]:,} rows")
            s3.metric("y_train", f"{len(y_train):,}")
            s4.metric("y_test",  f"{len(y_test):,}")
            st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].pie([len(X_train), len(X_test)], labels=["Train 80%","Test 20%"],
                        autopct="%1.1f%%", colors=["#4f8ef7","#f39c12"],
                        startangle=90, pctdistance=0.75, textprops={"fontsize":12})
            axes[0].set_title("Train / Test Split", fontweight="bold", fontsize=13)
            vc_tr = pd.Series(y_train).value_counts()
            axes[1].pie(vc_tr.values, labels=vc_tr.index, autopct="%1.1f%%",
                        colors=[COLORS["no"],COLORS["yes"]],
                        startangle=90, pctdistance=0.75, textprops={"fontsize":12})
            axes[1].set_title("Class Distribution in y_train", fontweight="bold", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with pp2:
            st.markdown("""
            <div class="card">
                <h4>🔡 One-Hot Encoding — <code>pd.get_dummies</code></h4>
                <pre><code>X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, dtype=int)
X_test_encoded  = pd.get_dummies(X_test,  columns=categorical_cols, dtype=int)
X_test_encoded  = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)</code></pre>
            </div>
            """, unsafe_allow_html=True)
            e1, e2, e3 = st.columns(3)
            e1.metric("Features before OHE", X_train.shape[1])
            e2.metric("Features after OHE",  X_train_encoded.shape[1])
            e3.metric("New dummy columns",    X_train_encoded.shape[1] - X_train.shape[1])
            ohe_cols = [c for c in X_train_encoded.columns
                        if any(c.startswith(cat+"_") for cat in categorical_cols)]
            st.dataframe(pd.DataFrame({"Encoded Column": ohe_cols}), use_container_width=True, height=300)

        with pp3:
            st.markdown("""
            <div class="card card-green">
                <h4>📏 Why Robust Scaler?</h4>
                <p>Scales using <b>IQR (Q1–Q3)</b> — insensitive to outliers unlike StandardScaler.</p>
                <pre><code>rs = RobustScaler()
X_train_sc = rs.fit_transform(X_train_encoded)
X_test_sc  = rs.transform(X_test_encoded)</code></pre>
            </div>
            """, unsafe_allow_html=True)
            sel_sc  = st.selectbox("Compare before / after scaling",
                                   X_train_encoded.columns.tolist(), key="sc_sel")
            col_idx = list(X_train_encoded.columns).index(sel_sc)
            before  = X_train_encoded[sel_sc].values
            after   = X_train_sc[:, col_idx]
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].hist(before, bins=35, color="#e74c3c", edgecolor="white", alpha=0.85)
            axes[0].set_title(f"Before Scaling — {sel_sc}", fontsize=13, fontweight="bold")
            axes[0].set_xlabel("Original Value", fontsize=11)
            axes[0].set_ylabel("Frequency", fontsize=11)
            axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            axes[0].grid(axis="y", linestyle="--", alpha=0.4)
            axes[1].hist(after, bins=35, color="#27ae60", edgecolor="white", alpha=0.85)
            axes[1].set_title(f"After Robust Scaling — {sel_sc}", fontsize=13, fontweight="bold")
            axes[1].set_xlabel("Scaled Value", fontsize=11)
            axes[1].set_ylabel("Frequency", fontsize=11)
            axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
            axes[1].grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with pp4:
            st.markdown("""
            <div class="card">
                <h4>🏷️ Label Encoding — Target <code>y</code></h4>
                <pre><code>le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)   # no=0  yes=1
y_test_enc  = le.transform(y_test)</code></pre>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Original": le.classes_,
                "Encoded":  list(range(len(le.classes_))),
            }), use_container_width=True)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            for ax, data, title in zip(
                axes,
                [pd.Series(y_train_enc).value_counts().sort_index(),
                 pd.Series(y_test_enc).value_counts().sort_index()],
                ["y_train_enc", "y_test_enc"],
            ):
                bars = ax.bar(["no (0)","yes (1)"], data.values,
                              color=[COLORS["no"],COLORS["yes"]],
                              edgecolor="white", width=0.5)
                vlabels_pp(ax, bars)
                ax.set_title(title, fontsize=13, fontweight="bold")
                ax.set_ylabel("Count", fontsize=11)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
                ax.grid(axis="y", linestyle="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with pp5:
            st.dataframe(pd.DataFrame({
                "Variable": ["X_train_sc","X_test_sc","y_train_enc","y_test_enc"],
                "Shape":    [str(X_train_sc.shape), str(X_test_sc.shape),
                             str(y_train_enc.shape), str(y_test_enc.shape)],
                "Type":     ["ndarray float64","ndarray float64","ndarray int64","ndarray int64"],
                "Note":     ["OHE + RobustScaler","OHE + same scaler no leakage","no=0 yes=1","no=0 yes=1"],
            }), use_container_width=True)
            st.markdown("""
            <div class="card card-green">
                <h4>✅ Preprocessing Pipeline — Summary</h4>
                <ol>
                    <li>Drop <code>default</code> column</li>
                    <li>Train-Test Split — 80/20, random_state=42</li>
                    <li>One-Hot Encoding on 9 categorical columns</li>
                    <li>Test columns reindexed to match train</li>
                    <li>RobustScaler — fit on train only</li>
                    <li>LabelEncoder on target — no=0, yes=1</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("❌ `bank-additional-full.csv` not found.")
    except ImportError as e:
        st.error(f"❌ {e} → pip install scikit-learn")


# ════════════════════════════════════════════════════════════
#  MODEL BUILDING
# ════════════════════════════════════════════════════════════
elif page == "Model Building":
    st.markdown('<p class="sec-title">🤖 Model Building & Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">All models trained — Base | SMOTE | ADASYN — compared with latency</p>', unsafe_allow_html=True)

    try:
        all_metrics     = joblib.load("models/all_metrics.joblib")
        best_info       = joblib.load("models/best_model_info.joblib")
        latency_results = joblib.load("models/latency_results.joblib")
    except FileNotFoundError:
        st.error("❌ Run models.py first.")
        st.stop()

    rows = []
    for m in all_metrics:
        rows.append({
            "Model":        m["model"],
            "Variant":      m["variant"],
            "Accuracy":     m["accuracy"],
            "Precision":    m["precision"],
            "Recall":       m["recall"],
            "F1 Score":     m["f1_score"],
            "ROC-AUC":      m["roc_auc"],
            "Latency (ms)": m.get("latency_ms", "-"),
        })
    full_df   = pd.DataFrame(rows)
    base_df   = full_df[full_df["Variant"] == "Base"].reset_index(drop=True)
    smote_df  = full_df[full_df["Variant"] == "SMOTE"].reset_index(drop=True)
    adasyn_df = full_df[full_df["Variant"] == "ADASYN"].reset_index(drop=True)

    MODEL_ORDER = ["Logistic Regression","Random Forest","XGBoost","Gradient Boosting","AdaBoost"]

    def highlight_max(s):
        is_max = s == s.max()
        return ["background-color:#d4edda; color:#155724; font-weight:bold" if v else "" for v in is_max]

    def metric_table(df):
        cols = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
        avail = [c for c in cols if c in df.columns]
        d = df.set_index("Model").drop(columns=["Variant"] + [c for c in ["Latency (ms)"] if c in df.columns], errors="ignore")
        return d.style.apply(highlight_max, subset=avail)

    def bar_chart(df, metric, title, color):
        fig, ax = plt.subplots(figsize=(13, 6))
        s = df.sort_values(metric, ascending=False)
        bars = ax.bar(range(len(s)), s[metric].values, color=color, edgecolor="white", width=0.55)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s["Model"].values, rotation=25, ha="right", fontsize=11)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, h+0.002, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim(min(s[metric].values)-0.02, 1.02)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Base Models", "🔵 SMOTE Models", "🟢 ADASYN Models",
        "⚖️ Full Comparison", "🏆 Best Model",
    ])

    with t1:
        st.markdown("#### 📊 All Models — No Balancing (Base)")
        st.markdown('<div class="card card-gold"><b>📌</b> Trained on original imbalanced data. High accuracy but low Recall on minority class.</div>', unsafe_allow_html=True)
        st.dataframe(metric_table(base_df), use_container_width=True)
        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
        metric_b = st.selectbox("Metric to visualise", ["ROC-AUC","Recall","F1 Score","Accuracy","Precision"], key="base_m")
        bar_chart(base_df, metric_b, f"{metric_b} — Base Models", "#4f8ef7")

    with t2:
        st.markdown("#### 🔵 All Models — SMOTE Balanced")
        st.markdown('<div class="card"><b>📌 SMOTE</b> — creates synthetic minority samples until both classes are equal.</div>', unsafe_allow_html=True)
        st.dataframe(metric_table(smote_df), use_container_width=True)
        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
        metric_s = st.selectbox("Metric to visualise", ["ROC-AUC","Recall","F1 Score","Accuracy","Precision"], key="smote_m")
        bar_chart(smote_df, metric_s, f"{metric_s} — SMOTE Models", "#27ae60")

    with t3:
        st.markdown("#### 🟢 All Models — ADASYN Balanced")
        st.markdown('<div class="card card-green"><b>📌 ADASYN</b> — focuses on harder minority samples near the decision boundary.</div>', unsafe_allow_html=True)
        st.dataframe(metric_table(adasyn_df), use_container_width=True)
        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
        metric_a = st.selectbox("Metric to visualise", ["ROC-AUC","Recall","F1 Score","Accuracy","Precision"], key="adasyn_m")
        bar_chart(adasyn_df, metric_a, f"{metric_a} — ADASYN Models", "#9b59b6")

    with t4:
        st.markdown("#### ⚖️ Full Comparison — All Models × All Variants")
        sort_col    = st.selectbox("Sort by", ["Recall","ROC-AUC","F1 Score","Accuracy","Precision"], key="full_sort")
        full_sorted = full_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

        def highlight_variant(row):
            c = {"Base":"background-color:#fff9e6","SMOTE":"background-color:#e8f4fd","ADASYN":"background-color:#eafaf1"}
            return [c.get(row["Variant"], "")] * len(row)

        st.dataframe(full_sorted.style.apply(highlight_variant, axis=1),
                     use_container_width=True, height=580)

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

        # grouped bar — Recall
        st.markdown("#### 📊 Recall — Base vs SMOTE vs ADASYN")
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(MODEL_ORDER))
        w = 0.25

        def safe_get(df, model, col):
            rows = df[df["Model"]==model][col].values
            return rows[0] if len(rows) > 0 else 0

        rb = [safe_get(base_df,   m, "Recall") for m in MODEL_ORDER]
        rs = [safe_get(smote_df,  m, "Recall") for m in MODEL_ORDER]
        ra = [safe_get(adasyn_df, m, "Recall") for m in MODEL_ORDER]

        b1 = ax.bar(x-w, rb, width=w, label="Base",   color="#4f8ef7", edgecolor="white")
        b2 = ax.bar(x,   rs, width=w, label="SMOTE",  color="#27ae60", edgecolor="white")
        b3 = ax.bar(x+w, ra, width=w, label="ADASYN", color="#9b59b6", edgecolor="white")
        for bars in [b1, b2, b3]:
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax.text(b.get_x()+b.get_width()/2, h+0.005, f"{h:.3f}",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right", fontsize=11)
        ax.set_ylabel("Recall", fontsize=12)
        ax.set_title("Recall — Base vs SMOTE vs ADASYN", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # latency chart
        st.markdown("#### ⚡ Prediction Latency — All Models (ms per prediction)")
        lat_df = pd.DataFrame([
            {"Model": k, "Latency (ms)": v}
            for k, v in latency_results.items()
        ]).sort_values("Latency (ms)", ascending=True).reset_index(drop=True)

        best_label = f"{best_info['name']}_{best_info['variant']}"
        bar_colors = ["#27ae60" if best_label in row["Model"] else "#aed6f1"
                      for _, row in lat_df.iterrows()]

        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.barh(range(len(lat_df)), lat_df["Latency (ms)"].values,
                       color=bar_colors, edgecolor="white", height=0.6)
        ax.set_yticks(range(len(lat_df)))
        ax.set_yticklabels(lat_df["Model"].values, fontsize=9)
        ax.set_xlabel("Latency (ms per prediction)", fontsize=12)
        ax.set_title("Prediction Latency — All Models (green = deployed model)",
                     fontsize=13, fontweight="bold", pad=12)
        xmax = lat_df["Latency (ms)"].max()
        for b in bars:
            w = b.get_width()
            ax.text(w + xmax*0.008, b.get_y()+b.get_height()/2,
                    f"{w:.4f} ms", va="center", ha="left",
                    fontsize=9, fontweight="bold")
        ax.set_xlim(0, xmax*1.2)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with t5:
        st.markdown("#### 🏆 Best Model — Logistic Regression (ADASYN)")
        st.markdown("""
        <div class="card card-gold">
            <b>📌 Why not the highest ROC-AUC model?</b><br>
            Gradient Boosting Base has the highest ROC-AUC (0.9476) but misses
            <b>414 actual subscribers</b> out of 935. This directly costs the bank revenue.
            LR ADASYN misses only <b>94 subscribers</b> — that is the right business choice.
        </div>
        """, unsafe_allow_html=True)

        bm = best_info["metrics"]
        cm = bm["confusion_matrix"]

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy",     bm["accuracy"])
        k2.metric("Precision",    bm["precision"])
        k3.metric("Recall",       bm["recall"])
        k4.metric("F1 Score",     bm["f1_score"])
        k5.metric("ROC-AUC",      bm["roc_auc"])

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"""
            <div class="card card-green">
                <h4>✅ Why Logistic Regression ADASYN?</h4>
                <ul>
                    <li>Highest <b>Recall = {bm['recall']}</b> — misses only
                        <b>{cm[1][0]} subscribers</b> out of {cm[1][0]+cm[1][1]}</li>
                    <li>Business goal is to <b>minimise False Negatives</b>
                        — every missed subscriber = lost revenue</li>
                    <li>ADASYN balancing focuses on the hardest minority samples
                        — model learns the difficult cases better</li>
                    <li>Fastest prediction latency among all models
                        — suitable for real-time deployment</li>
                    <li>Interpretable — coefficients show feature impact clearly</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class="card">
                <h4>📊 Missed Subscribers Comparison</h4>
                <table width="100%">
                    <tr style="background:#f5f5f5">
                        <td><b>Model</b></td>
                        <td><b>Recall</b></td>
                        <td><b>Missed</b></td>
                    </tr>
                    <tr style="background:#d4edda; font-weight:bold">
                        <td>✅ LR ADASYN (deployed)</td>
                        <td>0.8995</td>
                        <td>94</td>
                    </tr>
                    <tr>
                        <td>LR SMOTE</td>
                        <td>0.8759</td>
                        <td>116</td>
                    </tr>
                    <tr>
                        <td>GB ADASYN</td>
                        <td>0.8417</td>
                        <td>148</td>
                    </tr>
                    <tr>
                        <td>GB SMOTE</td>
                        <td>0.8267</td>
                        <td>162</td>
                    </tr>
                    <tr style="background:#fde8e8">
                        <td>GB Base (best ROC-AUC)</td>
                        <td>0.5283</td>
                        <td>414</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

        st.markdown("#### 🔢 Confusion Matrix — LR ADASYN")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No (0)","Yes (1)"],
                    yticklabels=["No (0)","Yes (1)"],
                    linewidths=0.5, ax=ax,
                    annot_kws={"size":14, "weight":"bold"})
        ax.set_title("Confusion Matrix — Logistic Regression (ADASYN)",
                     fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown(f"""
        <div class="card card-gold">
            <b>📌 Confusion Matrix Breakdown</b><br><br>
            ✅ <b>True Negative  (TN) = {cm[0][0]:,}</b> — correctly predicted will NOT subscribe<br>
            ⚠️ <b>False Positive (FP) = {cm[0][1]:,}</b> — predicted subscribe but didn't (extra calls)<br>
            ❌ <b>False Negative (FN) = {cm[1][0]:,}</b> — missed actual subscribers (revenue loss)<br>
            ✅ <b>True Positive  (TP) = {cm[1][1]:,}</b> — correctly predicted will subscribe<br><br>
            Only <b>{cm[1][0]} missed</b> out of {cm[1][0]+cm[1][1]} actual subscribers.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📋 Classification Report")
        st.code(bm["report"], language=None)

        st.markdown(f"""
        <div class="card card-green">
            <h4>✅ Final Conclusion</h4>
            <p><b>Logistic Regression with ADASYN</b> is the deployed model.
            Although Gradient Boosting Base achieved the highest ROC-AUC of 0.9476,
            it misses 414 actual subscribers vs only 94 missed by LR ADASYN.
            Since every missed subscriber is direct revenue loss for the bank,
            <b>Recall is the correct optimisation target</b> for this business problem.
            LR ADASYN achieves Recall of <b>{bm['recall']}</b> and has the lowest
            prediction latency — making it the right choice for real-time deployment.</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PREDICT
# ════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown('<p class="sec-title">🔮 Will This Client Subscribe?</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Enter the client details and the model predicts term deposit subscription likelihood.</p>', unsafe_allow_html=True)

    try:
        best_model = joblib.load("models/best_model.joblib")
        pre_data   = joblib.load("models/preprocessed_data.joblib")
        scaler     = pre_data["scaler"]
        le         = pre_data["label_encoder"]
        train_cols = pre_data["train_columns"]
    except FileNotFoundError:
        st.error("❌ Run preprocessing.py and models.py first.")
        st.stop()

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

    st.markdown("### 👤 Client Information")
    c1, c2, c3 = st.columns(3)

    with c1:
        age     = st.number_input("Age", min_value=18, max_value=100, value=35)
        job     = st.selectbox("Job", ["admin.","blue-collar","entrepreneur","housemaid",
                                       "management","retired","self-employed","services",
                                       "student","technician","unemployed","unknown"])
        marital = st.selectbox("Marital Status", ["married","single","divorced","unknown"])

    with c2:
        education = st.selectbox("Education Level", ["university.degree","high.school","basic.9y",
                                                      "professional.course","basic.4y","basic.6y",
                                                      "illiterate","unknown"])
        housing   = st.selectbox("Housing Loan?", ["yes","no","unknown"])
        loan      = st.selectbox("Personal Loan?", ["yes","no","unknown"])

    with c3:
        contact     = st.selectbox("Contact Type", ["cellular","telephone"])
        month       = st.selectbox("Last Contact Month", ["jan","feb","mar","apr","may","jun",
                                                          "jul","aug","sep","oct","nov","dec"])
        day_of_week = st.selectbox("Last Contact Day", ["mon","tue","wed","thu","fri"])

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
    st.markdown("### 📞 Campaign Details")
    d1, d2, d3, d4 = st.columns(4)

    with d1:
        duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=200,
                                   help="Longer calls generally mean higher interest")
    with d2:
        campaign = st.number_input("Contacts This Campaign", min_value=1, max_value=50, value=2)
    with d3:
        pdays    = st.number_input("Days Since Last Contact", min_value=0, max_value=999, value=999,
                                   help="999 = never contacted before")
    with d4:
        previous = st.number_input("Previous Campaign Contacts", min_value=0, max_value=50, value=0)

    poutcome = st.selectbox("Previous Campaign Outcome", ["nonexistent","failure","success"])

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
    st.markdown("### 🌍 Economic Indicators")
    st.markdown("""
    <div class="card card-gold">
        <b>📌 Note</b> — Macro-economic values. Defaults are the most recent values in the dataset.
        Expand to change only if you have updated data.
    </div>
    """, unsafe_allow_html=True)

    emp_var_rate   = -1.8
    cons_price_idx = 92.893
    cons_conf_idx  = -46.2
    euribor3m      = 1.299
    nr_employed    = 5099.1

    with st.expander("⚙️ Adjust Economic Indicators (optional)", expanded=False):
        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            emp_var_rate   = st.number_input("Emp. Variation Rate", min_value=-5.0, max_value=5.0, value=-1.8, step=0.1)
        with e2:
            cons_price_idx = st.number_input("Consumer Price Index", min_value=90.0, max_value=100.0, value=92.893, step=0.001, format="%.3f")
        with e3:
            cons_conf_idx  = st.number_input("Consumer Conf. Index", min_value=-60.0, max_value=0.0, value=-46.2, step=0.1)
        with e4:
            euribor3m      = st.number_input("Euribor 3M Rate", min_value=0.0, max_value=6.0, value=1.299, step=0.001, format="%.3f")
        with e5:
            nr_employed    = st.number_input("No. of Employees (K)", min_value=4900.0, max_value=5300.0, value=5099.1, step=0.1)

    st.markdown("<hr class='fancy'>", unsafe_allow_html=True)

    col_btn = st.columns([2, 1, 2])
    with col_btn[1]:
        predict_btn = st.button("🔮 Predict", use_container_width=True)

    if predict_btn:
        input_dict = {
            "age": age, "duration": duration, "campaign": campaign,
            "pdays": pdays, "previous": previous,
            "emp.var.rate": emp_var_rate, "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx, "euribor3m": euribor3m,
            "nr.employed": nr_employed,
            "job": job, "marital": marital, "education": education,
            "housing": housing, "loan": loan, "contact": contact,
            "month": month, "day_of_week": day_of_week, "poutcome": poutcome,
        }

        input_df      = pd.DataFrame([input_dict])
        cat_cols_pred = ["job","marital","education","housing","loan",
                         "contact","month","day_of_week","poutcome"]
        input_encoded = pd.get_dummies(input_df, columns=cat_cols_pred, dtype=int)
        input_encoded = input_encoded.reindex(columns=train_cols, fill_value=0)
        input_scaled  = scaler.transform(input_encoded)

        prediction  = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0]
        prob_no     = round(probability[0] * 100, 2)
        prob_yes    = round(probability[1] * 100, 2)

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
        st.markdown("### 📊 Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#27ae60,#1e8449);border-radius:16px;
                        padding:32px;text-align:center;margin:16px 0;
                        box-shadow:0 4px 15px rgba(39,174,96,0.3)">
                <h1 style="color:white;font-size:48px;margin:0">✅ YES</h1>
                <h3 style="color:#d5f5e3;margin:8px 0">This client is likely to subscribe to a term deposit</h3>
                <h2 style="color:white;margin:12px 0">Subscription Probability — {prob_yes}%</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#e74c3c,#c0392b);border-radius:16px;
                        padding:32px;text-align:center;margin:16px 0;
                        box-shadow:0 4px 15px rgba(231,76,60,0.3)">
                <h1 style="color:white;font-size:48px;margin:0">❌ NO</h1>
                <h3 style="color:#fadbd8;margin:8px 0">This client is unlikely to subscribe to a term deposit</h3>
                <h2 style="color:white;margin:12px 0">Non-Subscription Probability — {prob_no}%</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
        st.markdown("### 📈 Probability Breakdown")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(["Will NOT Subscribe","Will Subscribe"],
                [prob_no, prob_yes],
                color=["#e74c3c","#27ae60"], edgecolor="white", height=0.45)
        for i, v in enumerate([prob_no, prob_yes]):
            ax.text(v+0.5, i, f"{v}%", va="center", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 115)
        ax.set_xlabel("Probability (%)", fontsize=12)
        ax.set_title("Prediction Probability", fontsize=13, fontweight="bold", pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("<hr class='fancy'>", unsafe_allow_html=True)
        st.markdown("### 📋 Input Summary")
        summary_df = pd.DataFrame([{
            "Age": age, "Job": job, "Marital": marital, "Education": education,
            "Housing Loan": housing, "Personal Loan": loan, "Contact": contact,
            "Month": month, "Day": day_of_week, "Call Duration (s)": duration,
            "Campaign Contacts": campaign, "Pdays": pdays,
            "Previous": previous, "Prev. Outcome": poutcome,
        }]).T.reset_index()
        summary_df.columns = ["Feature", "Value"]
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="card" style="margin-top:16px">
            <small>
            🤖 <b>Model:</b> Logistic Regression (ADASYN) &nbsp;|&nbsp;
            🎯 <b>Optimised for:</b> Recall — minimise missed subscribers &nbsp;|&nbsp;
            📐 <b>Recall:</b> 0.8995 &nbsp;|&nbsp;
            📐 <b>ROC-AUC:</b> 0.9355
            </small>
        </div>
        """, unsafe_allow_html=True)