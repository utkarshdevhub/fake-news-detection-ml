from pathlib import Path
import joblib
import streamlit as st

MODEL_PATH = Path("models/best_model.joblib")

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)

st.title("📰 Fake News Detection")
st.caption("Machine Learning based text classification")

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. First place Fake.csv and True.csv in data/ "
        "and run: python train.py"
    )
    st.stop()

model = joblib.load(MODEL_PATH)

st.write(
    "Enter a news headline or article below. The model predicts which class "
    "the text most closely resembles based on its training data."
)

text = st.text_area(
    "News text",
    height=220,
    placeholder="Paste a headline or news article here..."
)

if st.button("Analyze News", type="primary"):
    if not text.strip():
        st.warning("Please enter some news text.")
    else:
        prediction = int(model.predict([text])[0])

        if prediction == 0:
            st.error("Prediction: FAKE")
        else:
            st.success("Prediction: REAL")

        if hasattr(model, "decision_function"):
            score = float(model.decision_function([text])[0])
            st.caption(
                f"Model decision score: {score:.3f}. "
                "This is not a factual probability."
            )

st.divider()
st.info(
    "Limitation: This system detects patterns learned from its dataset. "
    "It does not independently verify claims against authoritative sources."
)
