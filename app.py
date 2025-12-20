# app.py
# ============================================================
# WissAI — EEG Command Classification (Demo)
# Research proof-of-concept demo (recording-level inference).
#
# UI/UX upgraded (NO change to ML logic):
# - Tabs (Upload / Preview / Prediction / Model Info)
# - Status indicators + progress feel
# - Cleaner layout + cards
# - Expanders for technical details
# - Optional downloads (cleaned CSV, probability table)
# - NEW: "What the model did" explanation panel for examiners
# ============================================================

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from typing import Dict, Any, Optional, Tuple


# -----------------------------
# Page + light styling
# -----------------------------
st.set_page_config(page_title="WissAI — EEG Command Demo", layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.small-note { color: #6b7280; font-size: 0.9rem; }
.card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.7);
}
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  background: rgba(59,130,246,0.12);
  border: 1px solid rgba(59,130,246,0.2);
}
hr { margin: 1rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("## WissAI — EEG Command Classification (Demo)")
st.markdown('<span class="badge">Research proof-of-concept • recording-level inference</span>', unsafe_allow_html=True)
st.markdown(
    '<div class="small-note">'
    "This app demonstrates the end-to-end inference pipeline for examiners and supervisors. "
    "It is not a production system."
    "</div>",
    unsafe_allow_html=True
)

DEFAULT_EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
DEFAULT_REQUIRED_COLS = ["Timestamp"] + DEFAULT_EEG_CHANNELS


# -----------------------------
# Utilities
# -----------------------------
def bundle_keys(bundle: Any) -> list:
    return list(bundle.keys()) if isinstance(bundle, dict) else []


def find_first_key(d: Dict[str, Any], keys: list) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def load_bundle(model_path: str) -> Tuple[Dict[str, Any], Any, str]:
    obj = joblib.load(model_path)

    if not isinstance(obj, dict):
        return {"pipeline": obj}, obj, "pipeline"

    predictor_key = find_first_key(obj, [
        "svm_model", "pipeline", "model", "svm", "clf", "predictor", "estimator"
    ])
    if predictor_key is None:
        raise KeyError(
            "Could not find the trained model inside the bundle. "
            "Expected keys like 'svm_model'/'pipeline'/'model'. "
            f"Found keys: {list(obj.keys())}"
        )
    return obj, obj[predictor_key], predictor_key


def ensure_required_columns(df: pd.DataFrame, required_cols: list) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_muse_df(df: pd.DataFrame, eeg_channels: list, required_cols: list) -> pd.DataFrame:
    df = df[required_cols].copy()

    for c in eeg_channels:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df[eeg_channels] = df[eeg_channels].interpolate(limit_direction="both")
    df[eeg_channels] = df[eeg_channels].bfill().ffill()

    return df


# -----------------------------
# Feature extraction (MATCH notebook Cell 10)
# -----------------------------
def basic_stats(x: np.ndarray) -> list:
    mean = float(np.mean(x))
    std = float(np.std(x))
    mn = float(np.min(x))
    mx = float(np.max(x))
    rng = float(mx - mn)
    rms = float(np.sqrt(np.mean(x**2)))
    energy = float(np.sum(x**2) / (len(x) + 1e-8))
    return [mean, std, mn, mx, rng, rms, energy]


def fft_band_energy(x: np.ndarray, bands=((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5))) -> list:
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0)  # normalized
    total = np.sum(spec) + 1e-8

    feats = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(float(np.sum(spec[mask]) / total))
    return feats


def extract_recording_feature_vector(df_clean: pd.DataFrame, eeg_channels: list, signal_scaler) -> np.ndarray:
    vals = df_clean[eeg_channels].values.astype(np.float32)  # (T,4)
    vals_scaled = signal_scaler.transform(vals)              # (T,4)

    feat_vec = []
    for ch in range(vals_scaled.shape[1]):
        x = vals_scaled[:, ch]
        feat_vec.extend(basic_stats(x))
        feat_vec.extend(fft_band_energy(x))

    return np.array(feat_vec, dtype=np.float32).reshape(1, -1)


# -----------------------------
# Sidebar: Model loading
# -----------------------------
with st.sidebar:
    st.subheader("Model bundle (.joblib)")
    model_path = st.text_input("Path", value="wissai_svm_bundle.joblib")

    st.markdown("---")
    st.subheader("Display options")
    show_preview_rows = st.slider("Preview rows", 5, 50, 20)
    plot_points = st.slider("Plot samples", 200, 2000, 500, step=100)
    show_probability_table = st.checkbox("Show probability table (if available)", value=True)
    show_technical_details = st.checkbox("Show technical details", value=False)

bundle = None
model = None
predictor_key = None

try:
    with st.status("Loading model bundle...", expanded=False) as status:
        bundle, model, predictor_key = load_bundle(model_path)
        status.update(label="Model bundle loaded.", state="complete")
except Exception as e:
    st.error(str(e))
    st.stop()

cfg = bundle.get("config", {}) if isinstance(bundle, dict) else {}
eeg_channels = cfg.get("EEG_CHANNELS", bundle.get("EEG_CHANNELS", DEFAULT_EEG_CHANNELS))
required_cols = cfg.get("REQUIRED_COLS", bundle.get("REQUIRED_COLS", ["Timestamp"] + eeg_channels))

# TRAIN signal scaler (your key: pre_scaler)
signal_scaler = None
if isinstance(bundle, dict):
    scaler_key = find_first_key(bundle, ["pre_scaler", "scaler", "signal_scaler", "train_scaler", "window_scaler"])
    if scaler_key is not None:
        signal_scaler = bundle[scaler_key]

if signal_scaler is None:
    st.error(
        "Your bundle does not include the TRAIN signal scaler used in the notebook.\n\n"
        "Without it, the app cannot reproduce training preprocessing and predictions may collapse.\n\n"
        "Fix: re-save your bundle to include the train-only scaler (e.g., as 'pre_scaler').\n\n"
        f"Bundle keys found: {bundle_keys(bundle)}"
    )
    st.stop()

label_encoder = bundle.get("label_encoder", None) if isinstance(bundle, dict) else None


# -----------------------------
# Tabs layout
# -----------------------------
tab_upload, tab_preview, tab_prediction, tab_info = st.tabs(["Upload", "Preview", "Prediction", "Model Info"])

with tab_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload a Muse EEG CSV")
    st.caption("Required columns: Timestamp, TP9, AF7, AF8, TP10")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        st.info("Upload a CSV file to begin.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
        ensure_required_columns(df, required_cols)
    except Exception as e:
        st.error(f"CSV error: {e}")
        st.stop()

    st.success("File loaded successfully.")
    st.write(f"Rows: {len(df)}")

    try:
        df_clean = clean_muse_df(df, eeg_channels, required_cols)
    except Exception as e:
        st.error(f"Cleaning error: {e}")
        st.stop()

    st.session_state["df_raw"] = df
    st.session_state["df_clean"] = df_clean


with tab_preview:
    if "df_raw" not in st.session_state:
        st.info("Please upload a CSV first.")
        st.stop()

    df = st.session_state["df_raw"]
    df_clean = st.session_state["df_clean"]

    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data preview")
        st.dataframe(df.head(show_preview_rows), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Cleaned preview")
        st.dataframe(df_clean.head(show_preview_rows), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Signal quick plot")
    N = min(plot_points, len(df_clean))

    fig = plt.figure(figsize=(10, 4))
    for ch in eeg_channels:
        plt.plot(df_clean[ch].values[:N], label=ch)
    plt.title("EEG channels (preview)")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Downloads")
    st.download_button(
        "Download cleaned CSV",
        df_clean.to_csv(index=False).encode("utf-8"),
        file_name="wissai_cleaned.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)


with tab_prediction:
    if "df_clean" not in st.session_state:
        st.info("Please upload a CSV first.")
        st.stop()

    df_clean = st.session_state["df_clean"]

    # --- NEW: simple examiner-friendly explanation panel ---
    with st.expander("What the model did (simple explanation)", expanded=True):
        st.markdown(
            """
**This demo follows the same preprocessing used in the notebook (leakage-safe).**

**Step-by-step (recording-level inference):**
1) **Input:** You upload a Muse EEG CSV (Timestamp + TP9/AF7/AF8/TP10).  
2) **Cleaning:** Non-numeric values are converted; missing values are handled using interpolation and forward/back fill.  
3) **Train-only normalization:** The EEG signals are standardized using the **same scaler fitted only on the training data**.  
4) **Feature extraction:** A compact feature vector is computed from the full recording (per channel):
   - basic signal statistics (mean, std, min, max, range, RMS, energy)
   - compact frequency-band energy ratios (FFT-based)
5) **Prediction:** The trained **SVM classifier** predicts the most likely command label.  

**Important note for evaluation:**  
This is a **research proof-of-concept**. With limited recordings per class, generalization can be constrained, and the model may show overfitting. The demo is intended to transparently showcase the pipeline and the achieved results.
            """
        )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recording-level prediction")

    with st.spinner("Extracting features and predicting..."):
        X_feat = extract_recording_feature_vector(df_clean, eeg_channels, signal_scaler)
        pred_idx = int(model.predict(X_feat)[0])

        if label_encoder is not None:
            pred_label = str(label_encoder.inverse_transform([pred_idx])[0])
        else:
            pred_label = f"class_{pred_idx}"

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Predicted command", pred_label)

    with col2:
        st.write("Feature vector shape:", X_feat.shape)

    st.markdown("</div>", unsafe_allow_html=True)

    if show_probability_table and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_feat)[0]
        names = [str(x) for x in label_encoder.classes_] if label_encoder is not None else [f"class_{i}" for i in range(len(proba))]
        proba_df = pd.DataFrame({"class": names, "prob": proba}).sort_values("prob", ascending=False)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Class probabilities")
        st.dataframe(proba_df, use_container_width=True)

        st.download_button(
            "Download probabilities (CSV)",
            proba_df.to_csv(index=False).encode("utf-8"),
            file_name="wissai_probabilities.csv",
            mime="text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("This model does not expose predict_proba() or you disabled the probability table.")


with tab_info:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model info")

    st.write(f"Loaded predictor key: **{predictor_key}**")
    st.write("Expected columns:")
    st.json(required_cols)

    if show_technical_details:
        st.write("Bundle keys:")
        st.json(bundle_keys(bundle))

        st.write("Channels:")
        st.json(eeg_channels)

        detected_scaler = find_first_key(bundle, ["pre_scaler", "scaler", "signal_scaler", "train_scaler", "window_scaler"])
        st.write("Scaler key detected:")
        st.code(str(detected_scaler))

        if label_encoder is not None:
            st.write("Classes:")
            st.json([str(x) for x in label_encoder.classes_])

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Note: This demo is for academic presentation only (research proof-of-concept).")
