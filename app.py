import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ─────────────────────────────────────────────
# Title / Header / Sub-header
# ─────────────────────────────────────────────
st.title("🩺 Health Risk Predictor")
st.header("Student Illness Risk Assessment")
st.subheader("Using Logistic Regression — Enter your details in the sidebar to get a prediction")
st.markdown("---")

# ─────────────────────────────────────────────
# Helper: build a tiny synthetic dataset and
# train logistic regression from scratch
# (mirrors the notebook's approach)
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    """Train a logistic regression model on synthetic data that mirrors the notebook."""
    np.random.seed(42)
    n = 500

    age         = np.random.choice([19, 20, 21, 22, 23], n)
    stress      = np.random.choice([1, 2, 3, 4, 5], n)          # 1=low … 5=high
    sleep       = np.random.choice([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], n)
    screentime  = np.random.uniform(1, 12, n)
    water       = np.random.uniform(1, 4, n)   # litres

    # plausible illness probability
    logit = (
        -3
        + 0.1  * (age - 21)
        + 0.4  * stress
        - 0.3  * sleep
        + 0.2  * screentime
        - 0.5  * water
    )
    prob    = 1 / (1 + np.exp(-logit))
    illness = (np.random.rand(n) < prob).astype(float)

    X = np.column_stack([age, stress, sleep, screentime, water])
    y = illness.reshape(-1, 1)

    mean_X = np.mean(X, axis=0)
    std_X  = np.std(X, axis=0)
    X_sc   = (X - mean_X) / std_X
    X_aug  = np.concatenate([np.ones((X_sc.shape[0], 1)), X_sc], axis=1)

    def sigmoid(z):  return 1 / (1 + np.exp(-z))
    def cost(X, y, t): h = sigmoid(X @ t); m=len(y); return -(y*np.log(h+1e-9)+(1-y)*np.log(1-h+1e-9)).mean()
    def grad(X, y, t): h = sigmoid(X @ t); return X.T @ (h - y) / len(y)

    theta = np.zeros((X_aug.shape[1], 1))
    lr, costs = 0.1, []
    for _ in range(1000):
        costs.append(cost(X_aug, y, theta))
        theta -= lr * grad(X_aug, y, theta)

    # training accuracy
    preds = (sigmoid(X_aug @ theta) >= 0.5).astype(float)
    acc   = np.mean(preds == y)

    return theta, mean_X, std_X, acc, costs, X, y, illness


theta, mean_X, std_X, train_acc, costs, X_train, y_train, illness = train_model()


def predict(age, stress, sleep, screentime, water):
    x     = np.array([[age, stress, sleep, screentime, water]], dtype=float)
    x_sc  = (x - mean_X) / std_X
    x_aug = np.concatenate([np.ones((1, 1)), x_sc], axis=1)
    prob  = 1 / (1 + np.exp(-x_aug @ theta))
    return float(prob[0, 0])


# ─────────────────────────────────────────────
# Sidebar — collect user inputs
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=80)
st.sidebar.title("📋 Your Health Info")
st.sidebar.markdown("Fill in the fields below and click **Predict**.")

age = st.sidebar.selectbox(
    "🎂 Age",
    options=[19, 20, 21, 22, 23],
    index=1,
    help="Your current age",
)

stress = st.sidebar.slider(
    "😰 Stress Level (1 = Low, 5 = High)",
    min_value=1, max_value=5, value=3,
)

sleep_label = st.sidebar.selectbox(
    "😴 Daily Sleep Duration",
    options=["<3 hrs", "3-4 hrs", "4-5 hrs", "5-6 hrs", "6-7 hrs", "7-8 hrs", ">8 hrs"],
    index=4,
)
sleep_map = {"<3 hrs": 2.5, "3-4 hrs": 3.5, "4-5 hrs": 4.5,
             "5-6 hrs": 5.5, "6-7 hrs": 6.5, "7-8 hrs": 7.5, ">8 hrs": 8.5}
sleep = sleep_map[sleep_label]

screentime = st.sidebar.slider(
    "📱 Daily Screen Time (hours)",
    min_value=1.0, max_value=12.0, value=5.0, step=0.5,
)

water = st.sidebar.slider(
    "💧 Daily Water Intake (litres)",
    min_value=0.5, max_value=4.0, value=2.0, step=0.25,
)

predict_btn = st.sidebar.button("🔍 Predict My Risk", use_container_width=True)

# ─────────────────────────────────────────────
# Main layout — two columns
# ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# ── LEFT column: summary card ──
with col1:
    st.markdown("### 📝 Your Input Summary")
    summary_df = pd.DataFrame({
        "Feature":    ["Age", "Stress Level", "Sleep Duration", "Screen Time", "Water Intake"],
        "Your Value": [f"{age} yrs", f"{stress}/5", sleep_label,
                       f"{screentime} hrs/day", f"{water} L/day"],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("### 🏥 Model Info")
    st.info(
        f"**Algorithm:** Logistic Regression (from scratch)\n\n"
        f"**Features:** Age, Stress, Sleep, Screen Time, Water Intake\n\n"
        f"**Training Accuracy:** {train_acc*100:.1f}%"
    )

# ── RIGHT column: prediction result ──
with col2:
    st.markdown("### 🎯 Prediction Result")

    if predict_btn:
        prob = predict(age, stress, sleep, screentime, water)
        risk_pct = prob * 100

        if prob >= 0.5:
            st.error(f"⚠️ **High Risk of Illness**")
            st.metric(label="Illness Probability", value=f"{risk_pct:.1f}%", delta="Above threshold (50%)")
        else:
            st.success(f"✅ **Low Risk of Illness**")
            st.metric(label="Illness Probability", value=f"{risk_pct:.1f}%", delta="Below threshold (50%)", delta_color="off")

        # Gauge-style progress bar
        st.markdown("**Risk Meter:**")
        st.progress(prob)

        # Interpretation
        st.markdown("#### 💡 What this means")
        if risk_pct < 25:
            st.success("Your lifestyle looks healthy! Keep up the good habits.")
        elif risk_pct < 50:
            st.warning("Moderate risk. Consider improving sleep or reducing stress.")
        elif risk_pct < 75:
            st.warning("Elevated risk. Try getting more sleep and drinking more water.")
        else:
            st.error("High risk detected. Please consider consulting a healthcare professional.")

        # Tip cards
        st.markdown("#### 🌿 Personalized Tips")
        tips = []
        if sleep < 6: tips.append("😴 Aim for at least 7–8 hours of sleep per night.")
        if stress >= 4: tips.append("🧘 Practice stress management: meditation, exercise, or journaling.")
        if screentime > 7: tips.append("📵 Reduce screen time — take breaks every 30 minutes.")
        if water < 2: tips.append("💧 Drink at least 2–3 litres of water daily.")
        if not tips:
            tips.append("🎉 Great habits! Maintain your healthy routine.")
        for tip in tips:
            st.markdown(f"- {tip}")
    else:
        st.info("👈 Fill in your details in the sidebar and click **Predict My Risk**.")

st.markdown("---")

# ─────────────────────────────────────────────
# EDA Section (optional, collapsible)
# ─────────────────────────────────────────────
with st.expander("📊 Exploratory Data Analysis (EDA)", expanded=False):
    st.markdown("Visualizations based on the training dataset used to build the model.")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        st.markdown("**Illness Distribution**")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        labels = ["No Illness (0)", "Illness (1)"]
        counts = [int((illness == 0).sum()), int((illness == 1).sum())]
        ax1.bar(labels, counts, color=["#2ecc71", "#e74c3c"], edgecolor="white")
        ax1.set_ylabel("Count")
        ax1.set_title("Illness Class Distribution")
        st.pyplot(fig1)
        plt.close()

        st.markdown("**Stress Level Distribution**")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        stress_vals = X_train[:, 1]
        ax3.hist(stress_vals, bins=5, color="#3498db", edgecolor="white", rwidth=0.8)
        ax3.set_xlabel("Stress Level")
        ax3.set_ylabel("Count")
        ax3.set_title("Stress Level Distribution")
        st.pyplot(fig3)
        plt.close()

    with eda_col2:
        st.markdown("**Sleep Duration vs Illness**")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sleep_vals = X_train[:, 2]
        ax2.scatter(sleep_vals[illness == 0], np.random.uniform(0, 0.1, (illness == 0).sum()),
                    alpha=0.4, color="#2ecc71", label="No Illness", s=15)
        ax2.scatter(sleep_vals[illness == 1], np.random.uniform(0.9, 1.0, (illness == 1).sum()),
                    alpha=0.4, color="#e74c3c", label="Illness", s=15)
        ax2.set_xlabel("Sleep Duration (hrs)")
        ax2.set_yticks([0.05, 0.95]); ax2.set_yticklabels(["No", "Yes"])
        ax2.set_title("Sleep vs Illness")
        ax2.legend(fontsize=8)
        st.pyplot(fig2)
        plt.close()

        st.markdown("**Feature Correlation Heatmap**")
        df_corr = pd.DataFrame(X_train, columns=["Age","Stress","Sleep","Screentime","Water"])
        df_corr["Illness"] = illness
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax4, cbar=True, linewidths=0.5, annot_kws={"size": 7})
        ax4.set_title("Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("**Model Training — Cost Function Over Iterations**")
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    ax5.plot(costs, color="#9b59b6", linewidth=2)
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Cost")
    ax5.set_title("Logistic Regression Cost Function")
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)
    plt.close()

# ─────────────────────────────────────────────
# User Feedback Section (optional, collapsible)
# ─────────────────────────────────────────────
with st.expander("💬 User Feedback", expanded=False):
    st.markdown("Help us improve this tool by sharing your experience.")

    rating = st.radio(
        "How would you rate this prediction tool?",
        options=["⭐ Poor", "⭐⭐ Fair", "⭐⭐⭐ Good", "⭐⭐⭐⭐ Very Good", "⭐⭐⭐⭐⭐ Excellent"],
        horizontal=True,
    )

    accuracy_feedback = st.radio(
        "Did the prediction seem accurate for you?",
        options=["Yes, accurate", "Somewhat accurate", "Not accurate", "Not sure"],
        horizontal=True,
    )

    comments = st.text_area(
        "Any additional comments or suggestions?",
        placeholder="Type your feedback here...",
        height=100,
    )

    if st.button("📨 Submit Feedback"):
        if comments.strip() == "":
            st.warning("Please add a comment before submitting.")
        else:
            st.success(
                f"✅ Thank you for your feedback!\n\n"
                f"**Rating:** {rating}\n\n"
                f"**Accuracy:** {accuracy_feedback}\n\n"
                f"**Comments:** {comments}"
            )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px;'>"
    "🩺 Health Risk Predictor · Built with Streamlit · "
    "Logistic Regression from Scratch · For Educational Purposes Only"
    "</div>",
    unsafe_allow_html=True,
)
