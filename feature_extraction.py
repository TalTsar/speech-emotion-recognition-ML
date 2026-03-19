import os
import glob
import numpy as np
import librosa
import scipy.signal as signal
import config


# ==========================================
# WORKFLOW TOGGLE
# Set to True: Extracts everything (Do this ONCE to establish a sorted baseline).
# Set to False: Skips Wav2Vec2 and only extracts/updates acoustic features (FAST MODE).
# ==========================================
EXTRACT_EMBEDDINGS = True

if EXTRACT_EMBEDDINGS:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


def apply_bandpass_filter(data, sample_rate, lowcut=80.0, highcut=7500.0, order=5):
    """Applies a Butterworth bandpass filter to clean up the audio."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)


def extract_features(file_path, extractor=None, model=None):
    """Extracts Acoustic Features and (optionally) Deep Embeddings."""
    speech, sample_rate = librosa.load(file_path, sr=config.SAMPLE_RATE)

    # ====================================================
    # 0. STUDIO PREPROCESSING
    # ====================================================
    speech = apply_bandpass_filter(speech, sample_rate)
    max_amp = np.max(np.abs(speech))
    if max_amp > 0:
        speech = speech / max_amp

    # ====================================================
    # 1. ACOUSTIC FEATURE PIPELINE (The "Melody & Valence" Block)
    # ====================================================
    feature_blocks = []

    # 1. MFCCs (40 dimensions - captures raw vocal tract shape)
    mfccs = librosa.feature.mfcc(y=speech, sr=sample_rate, n_mfcc=40)
    feature_blocks.append(np.mean(mfccs.T, axis=0))

    # 2. Pitch & Prosody (Crucial for separating Happiness vs. Anger)
    f0 = librosa.yin(speech, fmin=50, fmax=500, sr=sample_rate)
    f0_voiced = f0[~np.isnan(f0)]

    if len(f0_voiced) > 1:
        pitch_mean = np.mean(f0_voiced)
        pitch_std = np.std(f0_voiced)  # How bouncy is the melody?
        pitch_range = np.max(f0_voiced) - np.min(f0_voiced)  # Extremes of the voice

        periods = 1.0 / f0_voiced
        jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
    else:
        pitch_mean, pitch_std, pitch_range, jitter = 0.0, 0.0, 0.0, 0.0

    feature_blocks.append(np.array([pitch_mean, pitch_std, pitch_range, jitter]))

    # 3. Harmonics-to-Noise Ratio (HNR Proxy) & Zero Crossing (Crucial for Anxiety)
    # Anxious voices tremble and introduce "noise" and breathiness.
    y_harmonic, y_percussive = librosa.effects.hpss(speech)
    rms_harm = np.mean(librosa.feature.rms(y=y_harmonic))
    rms_perc = np.mean(librosa.feature.rms(y=y_percussive))
    hnr_proxy = rms_harm / (rms_perc + 1e-6)

    zcr = np.mean(librosa.feature.zero_crossing_rate(speech))

    feature_blocks.append(np.array([hnr_proxy, zcr]))

    # 4. Energy & Shimmer
    frame_length = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)
    rms = librosa.feature.rms(y=speech, frame_length=frame_length, hop_length=hop_length)[0]

    shimmer = np.mean(np.abs(np.diff(rms))) / np.mean(rms) if len(rms) > 1 else 0.0
    feature_blocks.append(np.array([shimmer]))

    # 5. Spectral Flatness & Flux (Captures the "sharpness" of German Cold Anger)
    flatness = np.mean(librosa.feature.spectral_flatness(y=speech))
    flux = np.mean(librosa.onset.onset_strength(y=speech, sr=sample_rate))
    feature_blocks.append(np.array([flatness, flux]))

    # 6. Pause Ratio (Articulation Proxy)
    threshold = np.max(rms) * 0.1
    silent_frames = np.sum(rms < threshold)
    voiced_frames = np.sum(rms >= threshold)
    pause_ratio = silent_frames / (voiced_frames + 1e-6)
    feature_blocks.append(np.array([pause_ratio]))

    # Combine all acoustic features into one vector (now ~50 dimensions)
    acoustic_features = np.concatenate(feature_blocks)

    # ====================================================
    # 2. DEEP EMBEDDING PIPELINE (Conditional)
    # ====================================================
    embeddings = None
    if EXTRACT_EMBEDDINGS and extractor and model:
        input_values = extractor(speech, return_tensors="pt", sampling_rate=config.SAMPLE_RATE).input_values
        with torch.no_grad():
            outputs = model(input_values)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return acoustic_features, embeddings


def process_dataset():
    extractor, model = None, None
    if EXTRACT_EMBEDDINGS:
        print("Loading Wav2Vec2 Model (this may take a moment)...")
        extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        model.eval()
    else:
        print("🚀 FAST MODE ENABLED: Skipping Embeddings, extracting Acoustic Features only.")

    X_mfcc, y_excitation, X_emb_all, y_target_all, y_actor = [], [], [], [], []

    search_path = os.path.join(config.RAW_DATA_PATH, "**/*.wav")

    # CRITICAL FIX: sorted() ensures indices ALWAYS match across multiple runs
    audio_files = sorted(glob.glob(search_path, recursive=True))

    if not audio_files:
        print(f"Error: No .wav files found in {config.RAW_DATA_PATH}.")
        return

    is_test_mode = getattr(config, 'TEST_MODE', False)
    if is_test_mode:
        test_num = getattr(config, 'TEST_NUM_FILES', 40)
        audio_files = audio_files[:test_num]
        print(f"\nWARNING: RUNNING IN TEST MODE ({test_num} files)\n")

    print(f"Found {len(audio_files)} files to process. Starting extraction...")

    for i, file_path in enumerate(audio_files):
        if i % (10 if is_test_mode else 50) == 0:
            print(f"Processing file {i} / {len(audio_files)}")

        filename = os.path.basename(file_path).replace('.wav', '')
        if len(filename) != 7:
            continue

        emotion_code = filename[5]
        actor_id = int(filename[:2])

        if emotion_code not in config.EMOTION_DICT:
            continue

        # Extract features
        mfcc, emb = extract_features(file_path, extractor, model)
        exc_group = config.EXCITATION_MAPPING[emotion_code]

        X_mfcc.append(mfcc)
        y_excitation.append(exc_group)
        y_actor.append(actor_id)
        y_target_all.append(config.EMOTION_MAPPING[emotion_code])

        if EXTRACT_EMBEDDINGS:
            X_emb_all.append(emb)

    print("\nSaving features to disk...")
    # Always save the newly calculated acoustic features and labels
    np.save(os.path.join(config.FEATURE_SAVE_PATH, "X_mfcc.npy"), np.array(X_mfcc))
    np.save(os.path.join(config.FEATURE_SAVE_PATH, "y_excitation.npy"), np.array(y_excitation))
    np.save(os.path.join(config.FEATURE_SAVE_PATH, "y_target_all.npy"), np.array(y_target_all))
    np.save(os.path.join(config.FEATURE_SAVE_PATH, "y_actor.npy"), np.array(y_actor))

    # Only overwrite the heavy embeddings file if we actually calculated them
    if EXTRACT_EMBEDDINGS:
        np.save(os.path.join(config.FEATURE_SAVE_PATH, "X_emb_all.npy"), np.array(X_emb_all))
        print("Embeddings saved!")

    print("Feature extraction complete! Data is ready for training.")


if __name__ == "__main__":
    process_dataset()