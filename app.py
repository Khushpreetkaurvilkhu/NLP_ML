"""
Emotion Detector - Streamlit NLP app
Trains a TF-IDF + Logistic Regression model on train.txt (format: text;emotion)
and predicts the emotion of any text the user types in.

Run:  streamlit run app.py
"""
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Emotion Detector", page_icon=" ", layout="centered")

#  White theme 
st.markdown(
    """
    <style>
    .stApp, [data-testid="stHeader"], [data-testid="stSidebar"] {background:#FFFFFF;}
    [data-testid="stSidebar"] {border-right:1px solid #E5E7EB;}
    h1,h2,h3,p,label,span,div {color:#1F2933;}
    .result-card {border:1px solid #E5E7EB; border-radius:16px; padding:22px;
                  text-align:center; background:#FAFAFB; margin:12px 0;}
    .result-emoji {font-size:64px; line-height:1.1;}
    .result-label {font-size:30px; font-weight:700; margin-top:4px;}
    .result-conf {color:#6B7280; font-size:15px;}
    .bar-wrap {margin:6px 0;}
    .bar-label {display:flex; justify-content:space-between; font-size:14px;}
    .bar-bg {background:#EEF0F3; border-radius:8px; height:10px; overflow:hidden;}
    .bar-fill {background:#4F46E5; height:10px; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

EMOJI = {
    # pyrefly: ignore[parse-error]
    "joy" : " " , "sadness": " " , "anger": " ", "fear": "",
    "love": "", "surprise": "", "neutral": "",
}

# Small built-in dataset so the app still runs if train.txt is missing
FALLBACK = [
    ("i am so happy today everything is going great", "joy"),
    ("this is the best day of my life i feel wonderful", "joy"),
    ("i feel delighted and cheerful about the news", "joy"),
    ("i am thrilled i passed my exam", "joy"),
    ("i feel so lonely and miserable", "sadness"),
    ("i am heartbroken and cry every night", "sadness"),
    ("i feel down and hopeless today", "sadness"),
    ("i miss him so much it hurts", "sadness"),
    ("i am furious at how they treated me", "anger"),
    ("this makes me so angry i could scream", "anger"),
    ("i hate being lied to it is infuriating", "anger"),
    ("i feel irritated and annoyed by everything", "anger"),
    ("i am terrified of what might happen", "fear"),
    ("i feel anxious and scared about tomorrow", "fear"),
    ("the dark noise made me panic and tremble", "fear"),
    ("i am afraid something bad will happen", "fear"),
    ("i adore you and care about you deeply", "love"),
    ("i feel so much love and affection for my family", "love"),
    ("she is my sweetheart and i cherish her", "love"),
    ("i am fond of her and feel tender", "love"),
    ("i was shocked by the sudden surprise party", "surprise"),
    ("wow i did not expect that at all", "surprise"),
    ("i am amazed and stunned by the result", "surprise"),
    ("what a startling and unexpected turn", "surprise"),
]


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|@\w+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_lines(lines):
    rows = []
    for line in lines:
        line = line.strip()
        if ";" in line:
            text, label = line.rsplit(";", 1)
            rows.append((text.strip(), label.strip().lower()))
    return pd.DataFrame(rows, columns=["text", "label"])


@st.cache_data(show_spinner=False)
def load_data(path: str, uploaded_bytes: bytes | None):
    if uploaded_bytes is not None:
        lines = uploaded_bytes.decode("utf-8", errors="ignore").splitlines()
        return parse_lines(lines), "uploaded file"
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            df = parse_lines(f.readlines())
        if len(df) > 0:
            return df, path
    return pd.DataFrame(FALLBACK, columns=["text", "label"]), "built-in sample data"


@st.cache_resource(show_spinner="Training model...")
def train(df: pd.DataFrame):
    df = df.copy()
    df["clean"] = df["text"].map(clean)
    if len(df) < 200:  # tiny sample set: train on everything, no meaningful hold-out
        X_tr = X_te = df["clean"]
        y_tr = y_te = df["label"]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
        )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=5.0)),
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    report = pd.DataFrame(classification_report(y_te, pred, output_dict=True, zero_division=0)).T
    return model, acc, report


def bars(probs: pd.Series):
    html = ""
    for label, p in probs.sort_values(ascending=False).items():
        html += (
            f'<div class="bar-wrap"><div class="bar-label"><span>{EMOJI.get(label, "🙂")} '
            f'{label.title()}</span><span>{p*100:.1f}%</span></div>'
            f'<div class="bar-bg"><div class="bar-fill" style="width:{p*100:.1f}%"></div></div></div>'
        )
    st.markdown(html, unsafe_allow_html=True)


# Sidebar 
with st.sidebar:
    st.header(" Data")
    path = st.text_input("Training file path", "train.txt")
    up = st.file_uploader("...or upload train.txt (text;emotion)", type=["txt", "csv"])

df, source = load_data(path, up.getvalue() if up else None)
model, acc, report = train(df)

with st.sidebar:
    st.caption(f"Source: {source}")
    st.metric("Training samples", f"{len(df):,}")
    st.metric("Test accuracy", f"{acc*100:.1f}%")
    with st.expander("Class distribution"):
        st.bar_chart(df["label"].value_counts())
    with st.expander("Classification report"):
        st.dataframe(report.round(2))

# ---------- Main ----------
st.title(" Emotion Detector prediction model ")
st.write("Type a sentence and the NLP model will predict the emotion behind it.")

tab1, tab2 = st.tabs(["Single text", " Batch predict"])

with tab1:
    text = st.text_area("Your text", height=120,
                        placeholder="e.g. I can't believe how amazing this day turned out!")
    if st.button("Predict emotion", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            proba = model.predict_proba([clean(text)])[0]
            probs = pd.Series(proba, index=model.classes_)
            top = probs.idxmax()
            st.markdown(
                f'<div class="result-card"><div class="result-emoji">{EMOJI.get(top, " ")}</div>'
                f'<div class="result-label">{top.title()}</div>'
                f'<div class="result-conf">Confidence: {probs[top]*100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
            bars(probs)

with tab2:
    st.write("Paste one sentence per line.")
    multi = st.text_area("Lines", height=150, key="multi")
    if st.button("Predict all"):
        lines = [l for l in multi.splitlines() if l.strip()]
        if not lines:
            st.warning("Enter at least one line.")
        else:
            P = model.predict_proba([clean(l) for l in lines])
            out = pd.DataFrame({
                "text": lines,
                "emotion": [model.classes_[i] for i in P.argmax(1)],
                "confidence %": (P.max(1) * 100).round(1),
            })
            st.dataframe(out, use_container_width=True)
            st.download_button("Download CSV", out.to_csv(index=False), "predictions.csv")
