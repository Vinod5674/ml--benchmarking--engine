import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

# Page Configuration
st.set_page_config(
    page_title="Enterprise ML Benchmarking Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fast Data Loading with Caching
@st.cache_data
def load_csv_data(file):
    return pd.read_csv(file)

st.title("⚡ Enterprise ML Benchmarking & Inference Engine")

# Global State Initialization
if "processed" not in st.session_state:
    st.session_state.processed = False

with st.sidebar:
    st.header("⚙️ 1. Dataset Upload")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    
    if st.button("🔄 Reset App State"):
        st.session_state.clear()
        st.rerun()

if uploaded_file:
    df = load_csv_data(uploaded_file)

    # LAZY LOADING TABS
    selected_tab = st.radio(
        "Navigation",
        ["📋 Profile & Cleaning", "📊 EDA Studio", "⚙️ Pipeline Config", "🏆 Model Leaderboard", "🔮 Live Inference"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # ---------------- 1. PROFILE & CLEANING ----------------
    if selected_tab == "📋 Profile & Cleaning":
        st.subheader("Data Overview & Structural Quality")
        target_col = st.selectbox("Select Target Variable (Y)", df.columns, index=len(df.columns)-1)
        st.session_state["target_col"] = target_col

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Cells", df.isnull().sum().sum())
        c4.metric("Duplicates", df.duplicated().sum())

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Dataset Head")
            st.dataframe(df.head(5), use_container_width=True)
        with col_right:
            st.markdown("##### Column Data Types")
            st.dataframe(pd.DataFrame(df.dtypes, columns=["DataType"]).astype(str), use_container_width=True)

    # ---------------- 2. EDA STUDIO ----------------
    elif selected_tab == "📊 EDA Studio":
        st.subheader("Exploratory Data Analysis")
        target_col = st.session_state.get("target_col", df.columns[-1])

        e_col1, e_col2 = st.columns(2)
        plot_df = df.sample(min(len(df), 2000), random_state=42) if len(df) > 2000 else df

        with e_col1:
            fig_target = px.histogram(plot_df, x=target_col, color=target_col, title=f"Target Distribution ({target_col})", template="plotly_white")
            st.plotly_chart(fig_target, use_container_width=True)

        with e_col2:
            num_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) > 1:
                corr = plot_df[num_cols].corr()
                fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="Viridis", title="Correlation Heatmap")
                st.plotly_chart(fig_corr, use_container_width=True)

    # ---------------- 3. PIPELINE CONFIG ----------------
    elif selected_tab == "⚙️ Pipeline Config":
        st.subheader("Configure Preprocessing & Hyperparameters")
        target_col = st.session_state.get("target_col", df.columns[-1])

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            num_impute = st.selectbox("Numeric Imputation", ["median", "mean"])
            scale_opt = st.selectbox("Feature Scaling", ["standard", "minmax", "none"])
            cat_opt = st.selectbox("Categorical Encoding", ["onehot", "label"])

        with col_p2:
            selected_models = st.multiselect("Select Algorithms", ["Logistic Regression", "Decision Tree", "Random Forest"], default=["Logistic Regression", "Random Forest"])
            hyperparams = {}
            if "Logistic Regression" in selected_models:
                hyperparams["Logistic Regression"] = {"C": st.slider("Logistic Regression C", 0.01, 10.0, 1.0)}
            if "Decision Tree" in selected_models:
                hyperparams["Decision Tree"] = {"max_depth": st.slider("Decision Tree Max Depth", 1, 20, 5)}
            if "Random Forest" in selected_models:
                hyperparams["Random Forest"] = {
                    "n_estimators": st.slider("Random Forest Estimators", 10, 100, 30, step=10),
                    "max_depth": st.slider("Random Forest Max Depth", 1, 20, 5)
                }

        if st.button("🚀 Execute Pipeline & Train Models", type="primary"):
            with st.spinner("Processing Data & Training Models..."):
                dp = DataProcessor()
                X_proc, y_proc, feat_names = dp.preprocess_data(
                    df, target_col, num_strategy=num_impute, scale_method=scale_opt, categorical_action=cat_opt
                )

                # 🚨 Strict Filter: Remove any leftover One-Hot Surname/ID columns
                clean_features = [f for f in feat_names if not f.startswith("Surname_") and not f.lower().startswith("rownumber") and not f.lower().startswith("customerid")]
                X_proc = X_proc[clean_features]

                mt = ModelTrainer()
                results_df, fitted_models, X_tr, X_te, y_tr, y_te = mt.train_and_evaluate(
                    X_proc, y_proc, selected_models, hyperparams
                )

                st.session_state.results_df = results_df
                st.session_state.fitted_models = fitted_models
                st.session_state.data_processor = dp
                st.session_state.model_trainer = mt
                st.session_state.feature_names = clean_features
                st.session_state.X_test = X_te
                st.session_state.y_test = y_te
                st.session_state.processed = True
                st.success("Training Complete! Move to 'Model Leaderboard'.")

    # ---------------- 4. MODEL LEADERBOARD ----------------
    elif selected_tab == "🏆 Model Leaderboard":
        if not st.session_state.processed:
            st.info("👈 Please execute the pipeline in '⚙️ Pipeline Config' tab first.")
        else:
            results_df = st.session_state.results_df
            fitted_models = st.session_state.fitted_models
            feat_names = st.session_state.feature_names

            st.subheader("🏆 Algorithm Benchmarking Leaderboard")
            st.dataframe(results_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

            fig_bar = px.bar(
                results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"]),
                x="Model", y="value", color="variable", barmode="group", title="Cross-Metric Model Comparison", template="plotly_white"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            model_to_inspect = st.selectbox("Inspect Feature Importance", list(fitted_models.keys()))
            mt = st.session_state.model_trainer
            df_imp = mt.get_feature_importance(fitted_models[model_to_inspect], feat_names)

            if df_imp is not None:
                fig_imp = px.bar(df_imp.head(10), x="Importance", y="Feature", orientation="h", title="Top 10 Features", template="plotly_white")
                fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------- 5. LIVE INFERENCE ----------------
    elif selected_tab == "🔮 Live Inference":
        if not st.session_state.processed:
            st.info("👈 Please execute the pipeline in '⚙️ Pipeline Config' tab first.")
        else:
            st.subheader("🔮 Real-Time Single Prediction Interface")
            feat_names = st.session_state.feature_names
            top_model = st.session_state.fitted_models[st.session_state.results_df.iloc[0]["Model"]]

            st.info(f"Showing **{len(feat_names)}** active model inputs.")

            input_data = {}
            col_list = st.columns(3)
            for idx, feature in enumerate(feat_names):
                with col_list[idx % 3]:
                    input_data[feature] = st.number_input(f"Value for {feature}", value=0.0)

            if st.button("Generate Live Prediction", type="primary"):
                input_df = pd.DataFrame([input_data])
                prediction = top_model.predict(input_df)[0]
                st.success(f"Prediction Output Label: `{prediction}`")

else:
    st.info("👈 Upload a CSV dataset from sidebar to get started.")