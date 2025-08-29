import streamlit as st
import glob
import re
import pickle
import numpy as np
import tensorflow as tf

TOKENIZER_PATH = r'deployments/x_tokenizer.pkl'
MODEL_PATHS = glob.glob(r'deployments/*.keras')  # 🔹 all Keras models


def main():
    st.set_page_config(
        page_title="Duplicate Similar Questions",
        page_icon="⁉",
        layout="wide",
    )

    # Load once into session_state
    if "tokenizer" not in st.session_state or "models" not in st.session_state:
        st.session_state["tokenizer"], st.session_state["models"] = load()

    st.title("Duplicate Questions Similarity Learning")

    with st.container():
        cols = st.columns(2)

        q1_str = cols[0].text_input("Question 1")
        if q1_str and len(str(q1_str)) < 4:
            st.warning("Q1 is too short")
            st.session_state["q1"] = None
        else:
            st.session_state["q1"] = q1_str

        q2_str = cols[1].text_input("Question 2")
        if q2_str and len(str(q2_str)) < 4:
            st.warning("Q2 is too short")
            st.session_state["q2"] = None
        else:
            st.session_state["q2"] = q2_str

        # Model Picker
        model_name = st.selectbox("Select Model", MODEL_PATHS)
        st.session_state["model_name"] = model_name

        if st.button("Check Similar", disabled=not st.session_state["q2"] or not st.session_state["q1"]):
            predict()


@st.cache_resource
def load():
    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Load keras models
    models = {}
    for path in MODEL_PATHS:
        models[path] = tf.keras.models.load_model(path,safe_mode=False)

    return tokenizer, models


def predict():
    q1 = clean_str(st.session_state["q1"])
    q2 = clean_str(st.session_state["q2"])

    tokenizer = st.session_state["tokenizer"]
    models = st.session_state["models"]
    model_path = st.session_state["model_name"]
    model = models[model_path]

    # Tokenize
    seq1 = tokenizer.texts_to_sequences([q1])
    seq2 = tokenizer.texts_to_sequences([q2])

    # ⚠️ Ensure fixed length (must match INPUT_SHAPE used in training)
    MAX_LEN = 50  # 🔹 change to whatever you trained with
    seq1 = tf.keras.preprocessing.sequence.pad_sequences(seq1, maxlen=MAX_LEN, padding="post")
    seq2 = tf.keras.preprocessing.sequence.pad_sequences(seq2, maxlen=MAX_LEN, padding="post")

    # Prepare input (batch_size=1, 2 sequences, sequence_length)
    X = np.stack([seq1[0], seq2[0]], axis=0)[None, ...].astype(np.float32)

    # Run inference
    prediction = model.predict(X, verbose=0)[0][0]

    # Show result
    st.success(f"Predicted Similarity: {prediction:.4f}")
    if prediction >= 0.5:
        st.write("✅ The questions are likely **duplicates/similar**.")
    else:
        st.write("❌ The questions are likely **different**.")


def clean_str(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-zA-Z0-9]", " ", sentence)
    sentence = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", sentence)  # 12cm -> 12 cm
    sentence = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", sentence)  # cm12 -> cm 12
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


if __name__ == '__main__':
    main()
