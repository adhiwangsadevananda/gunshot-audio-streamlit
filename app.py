# app.py
import streamlit as st
import librosa
import numpy as np
import pickle

# Load model, scaler, selector, dan label encoder
with open("model.pkl", "rb") as f_model:
    model = pickle.load(f_model)

with open("scaler.pkl", "rb") as f_scaler:
    scaler = pickle.load(f_scaler)

with open("selector.pkl", "rb") as f_fs:
    feature_selector = pickle.load(f_fs)

with open("label_encoder.pkl", "rb") as f_le:
    label_encoder = pickle.load(f_le)

# Fungsi ekstraksi fitur
def extract_features_rich(audio, sr=22050):
    import scipy.stats as stats

    features = []

    # Time domain
    features += [
        np.mean(audio), np.std(audio), np.max(audio), np.min(audio),
        np.var(audio), np.median(audio), np.percentile(audio, 25),
        np.percentile(audio, 75), np.percentile(audio, 10), np.percentile(audio, 90),
        np.sum(np.abs(audio)), np.sqrt(np.mean(audio**2)), len(audio),
        stats.skew(audio), stats.kurtosis(audio)
    ]

    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features += [np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)]

    envelope = np.abs(audio)
    features += [np.mean(envelope), np.std(envelope), np.max(envelope)]

    peak_idx = np.argmax(np.abs(audio))
    attack_time = peak_idx / len(audio)
    features.append(attack_time)

    if peak_idx < len(audio) - 1:
        decay_part = audio[peak_idx:]
        decay_slope = np.polyfit(range(len(decay_part)), decay_part, 1)[0]
        features.append(decay_slope)
    else:
        features.append(0)

    # Frequency domain
    fft = np.fft.fft(audio)
    magnitude = np.abs(fft)
    features += [
        np.mean(magnitude), np.std(magnitude), np.max(magnitude),
        np.var(magnitude), np.median(magnitude)
    ]

    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features += [np.mean(spectral_centroids), np.std(spectral_centroids),
                 np.max(spectral_centroids), np.min(spectral_centroids)]

    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features += [np.mean(spectral_rolloff), np.std(spectral_rolloff),
                 np.max(spectral_rolloff), np.min(spectral_rolloff)]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features += [np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                 np.max(spectral_bandwidth), np.min(spectral_bandwidth)]

    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)[0]
    features += [np.mean(spectral_contrast), np.std(spectral_contrast)]

    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
    features += [np.mean(spectral_flatness), np.std(spectral_flatness)]

    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    for i in range(tonnetz.shape[0]):
        features += [np.mean(tonnetz[i]), np.std(tonnetz[i])]

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    for i in range(13):
        features += [
            np.mean(mfcc[i]), np.std(mfcc[i]), np.max(mfcc[i]),
            np.min(mfcc[i]), np.median(mfcc[i]), np.var(mfcc[i])
        ]
    for i in range(13):
        features += [np.mean(mfcc_delta[i]), np.std(mfcc_delta[i])]
    for i in range(13):
        features += [np.mean(mfcc_delta2[i]), np.std(mfcc_delta2[i])]

    return np.array(features)

# UI
st.title("🔫 Gunshot Audio Classifier")
st.markdown("Upload file audio `.wav` untuk mengetahui jenis senjata yang digunakan.")

uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Tombol untuk mulai klasifikasi
    if st.button("🎯 Klasifikasikan"):
        try:
            # Load audio
            audio, sr = librosa.load(uploaded_file, sr=22050)

            # Ekstraksi fitur
            features = extract_features_rich(audio, sr=sr).reshape(1, -1)
            features_scaled = scaler.transform(features)
            features_selected = feature_selector.transform(features_scaled)

            # Prediksi
            pred = model.predict(features_selected)[0]
            prob = model.predict_proba(features_selected)[0]
            pred_label = label_encoder.inverse_transform([pred])[0]
            pred_confidence = prob[pred] * 100

            # Tampilkan hasil
            st.subheader("🎯 Hasil Prediksi:")
            st.success(f"Senjata terdeteksi: **{pred_label}**")
            st.info(f"Akurasi klasifikasi: **{pred_confidence:.2f}%**")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses: {e}")
