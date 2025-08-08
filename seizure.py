import streamlit as st
import numpy as np
import pywt
import matplotlib.pyplot as plt
import pyedflib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns

st.title("🧠 EEG Seizure Detection using LSTM + Wavelet Transform")


# --- Load EEG ---
def load_edf(file, channel=0):
    f = pyedflib.EdfReader(file)
    signal = f.readSignal(channel)
    labels = f.getSignalLabels()
    f._close()
    return signal, labels


eeg_signal, channels = load_edf("chb01_03.edf")
st.success(f"Loaded EEG with {len(channels)} channels. Using channel 0: {channels[0]}")

# --- Show sample EEG plot ---
st.subheader("📈 Sample EEG Signal")
fig, ax = plt.subplots()
ax.plot(eeg_signal[:1000])
ax.set_title("First 1000 samples of EEG")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")
st.pyplot(fig)


# --- Wavelet feature extraction ---
def extract_wavelet_features(signal, wavelet="db4", level=4, window_size=256, step=128):
    features = []
    for start in range(0, len(signal) - window_size + 1, step):
        segment = signal[start : start + window_size]
        coeffs = pywt.wavedec(segment, wavelet, level=level)
        feat = [np.mean(np.abs(c)) for c in coeffs]
        features.append(feat)
    return np.array(features)


# --- Extract features from full signal ---
st.subheader("⚙️ Extracting Wavelet Features...")
features = extract_wavelet_features(eeg_signal)
features = StandardScaler().fit_transform(features)

# --- Create labels (dummy seizure in last 30%) ---
st.info("Labeling last 30% of the samples as 'seizure' for demonstration.")
labels = np.zeros(len(features))
labels[-int(len(features) * 0.3) :] = 1

# --- Wavelet Coefficients Visualization ---
st.subheader("🌊 Wavelet Coefficients of Sample Window")
sample_window = eeg_signal[100:356]
coeffs = pywt.wavedec(sample_window, "db4", level=4)
fig3, axs = plt.subplots(5, 1, figsize=(10, 8))
axs[0].plot(sample_window)
axs[0].set_title("Original EEG Window")
for i in range(4):
    axs[i + 1].plot(coeffs[i])
    axs[i + 1].set_title(f"Wavelet Coeff {i+1}")
plt.tight_layout()
st.pyplot(fig3)

# --- Format for LSTM ---
timesteps = 5
samples = len(features) // timesteps
X = features[: samples * timesteps].reshape(samples, timesteps, -1)
y = labels[: samples * timesteps].reshape(samples, timesteps, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --- Model Training ---
st.subheader("🔁 Model Training")
if st.button("🚀 Train the Model"):
    with st.spinner("Training LSTM model..."):
        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.3),
                LSTM(32, return_sequences=True),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.fit(
            X_train, y_train, epochs=5, batch_size=16, validation_split=0.2, verbose=0
        )
        st.session_state.model = model
        st.success("✅ Model training completed.")

        # Predictions
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        y_true = y_test.flatten()
        y_pred_flat = y_pred.flatten()

        acc = accuracy_score(y_true, y_pred_flat)
        st.subheader("📊 Model Evaluation")
        st.write(f"**Accuracy:** {acc:.2f}")

        # Plot: Prediction vs Actual
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(y_true, label="True")
        ax2.plot(y_pred_flat, label="Predicted", alpha=0.7)
        ax2.set_title("Predicted vs Actual Seizure Labels")
        ax2.legend()
        st.pyplot(fig2)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred_flat)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Seizure", "Seizure"],
            yticklabels=["Non-Seizure", "Seizure"],
            ax=ax_cm,
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("🧾 Confusion Matrix")
        st.pyplot(fig_cm)

# --- Predict from Raw EEG ---
st.subheader("🧠 Predict from Raw EEG Signal")
required_points = 128 * (timesteps - 1) + 256
st.info(
    f"ℹ️ Please enter at least {required_points} EEG points to allow {timesteps} timesteps."
)

user_input = st.text_area("Enter raw EEG values (comma-separated)", height=150)

if st.button("📈 Predict from Raw EEG"):
    try:
        values = [float(val.strip()) for val in user_input.split(",") if val.strip()]
        if len(values) < required_points:
            st.warning(f"⚠️ Not enough data. Enter at least {required_points} values.")
        else:
            raw_signal = np.array(values)
            feature_vector = extract_wavelet_features(raw_signal)

            if feature_vector.shape[0] < timesteps:
                st.error(
                    "❌ Not enough segments after wavelet transform. Enter longer EEG input."
                )
            else:
                # Pad or trim to match required timesteps
                if feature_vector.shape[0] > timesteps:
                    feature_vector = feature_vector[:timesteps]
                elif feature_vector.shape[0] < timesteps:
                    pad = np.zeros(
                        (timesteps - feature_vector.shape[0], feature_vector.shape[1])
                    )
                    feature_vector = np.vstack([feature_vector, pad])

                # Reshape and predict
                reshaped = feature_vector.reshape(1, timesteps, -1)
                prediction = st.session_state.model.predict(reshaped)
                predicted_probs = prediction[0].flatten()
                predicted_classes = (predicted_probs > 0.5).astype(int)

                if np.any(predicted_classes == 1):
                    st.success("✅ Seizure detected in input segment.")
                else:
                    st.success("✅ No seizure detected in input segment.")

                st.line_chart(predicted_probs)
    except Exception as e:
        st.error(f"❌ Error: {e}")
