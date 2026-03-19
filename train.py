import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import config

EMOTION_DECODER = {
    0: 'Anger', 1: 'Boredom', 2: 'Disgust', 3: 'Anxiety/Fear',
    4: 'Happiness', 5: 'Sadness', 6: 'Neutral'
}


def main():
    print("Loading all datasets from disk...")
    # 1. Load the Master Lists
    X_mfcc = np.load(os.path.join(config.FEATURE_SAVE_PATH, "X_mfcc.npy"))
    X_emb = np.load(os.path.join(config.FEATURE_SAVE_PATH, "X_emb_all.npy"))  # These are just the 768 W2V2 embeddings
    y_exc = np.load(os.path.join(config.FEATURE_SAVE_PATH, "y_excitation.npy"))
    y_target = np.load(os.path.join(config.FEATURE_SAVE_PATH, "y_target_all.npy"))
    y_actor = np.load(os.path.join(config.FEATURE_SAVE_PATH, "y_actor.npy"))

    k = 5
    gkf = GroupKFold(n_splits=k)

    all_y_true = []
    all_pipeline_preds = []
    error_log = []

    all_exc_true, all_exc_pred = [], []
    all_low_true, all_low_pred = [], []
    all_high_true, all_high_pred = [], []

    print(f"\n--- Starting True Pipeline {k}-Fold Cross-Validation ---")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_mfcc, y_target, groups=y_actor), 1):
        print(f"Processing Fold {fold}...")

        # ==========================================
        # STEP 1: SPLIT DATA FOR THIS FOLD
        # ==========================================
        X_mfcc_train, X_mfcc_test = X_mfcc[train_idx], X_mfcc[test_idx]
        X_emb_train, X_emb_test = X_emb[train_idx], X_emb[test_idx]
        y_exc_train, y_exc_test = y_exc[train_idx], y_exc[test_idx]
        y_target_train, y_target_test = y_target[train_idx], y_target[test_idx]
        y_actor_test = y_actor[test_idx]

        # ==========================================
        # STEP 2: TRAIN THE MODELS (Safe PCA & Fusion)
        # ==========================================
        # A. Gatekeeper (L1) - Uses ONLY Acoustic Features
        scaler_L1 = StandardScaler()
        X_mfcc_train_scaled = scaler_L1.fit_transform(X_mfcc_train)
        gatekeeper = SVC(kernel='rbf', C=10, gamma='scale')
        gatekeeper.fit(X_mfcc_train_scaled, y_exc_train)

        # COMPRESS THE EMBEDDINGS
        pca_emb = PCA(n_components=50, random_state=42)
        X_emb_train_pca = pca_emb.fit_transform(X_emb_train)

        # COMBINE FEATURES
        # New feature vector size: 100 dense dimensions
        X_combined_train = np.hstack((X_mfcc_train_scaled, X_emb_train_pca))

        # B. Low Specialist (L2)
        low_mask_train = (y_exc_train == 0)
        scaler_L2_low = StandardScaler()
        X_low_train_scaled = scaler_L2_low.fit_transform(X_combined_train[low_mask_train])
        low_specialist = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
        low_specialist.fit(X_low_train_scaled, y_target_train[low_mask_train])

        # C. High Specialist (L2)
        high_mask_train = (y_exc_train == 1)
        scaler_L2_high = StandardScaler()
        X_high_train_scaled = scaler_L2_high.fit_transform(X_combined_train[high_mask_train])
        high_specialist = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
        high_specialist.fit(X_high_train_scaled, y_target_train[high_mask_train])

        # ==========================================
        # STEP 3: PIPELINE INFERENCE
        # ==========================================
        # 1. Scale test acoustics and route
        X_mfcc_test_scaled = scaler_L1.transform(X_mfcc_test)
        pred_exc_routes = gatekeeper.predict(X_mfcc_test_scaled)

        # 2. Compress test embeddings using the trained PCA
        X_emb_test_pca = pca_emb.transform(X_emb_test)

        # 3. Combine test features exactly like we did for training
        X_combined_test = np.hstack((X_mfcc_test_scaled, X_emb_test_pca))

        # Track Individual Performance
        all_exc_true.extend(y_exc_test)
        all_exc_pred.extend(pred_exc_routes)

        low_mask_test = (y_exc_test == 0)
        if np.any(low_mask_test):
            X_low_test_scaled = scaler_L2_low.transform(X_combined_test[low_mask_test])
            all_low_pred.extend(low_specialist.predict(X_low_test_scaled))
            all_low_true.extend(y_target_test[low_mask_test])

        high_mask_test = (y_exc_test == 1)
        if np.any(high_mask_test):
            X_high_test_scaled = scaler_L2_high.transform(X_combined_test[high_mask_test])
            all_high_pred.extend(high_specialist.predict(X_high_test_scaled))
            all_high_true.extend(y_target_test[high_mask_test])

        # Actual Pipeline Routing
        fold_preds = []
        for i in range(len(test_idx)):
            sample_combined = X_combined_test[i].reshape(1, -1)

            if pred_exc_routes[i] == 0:
                final_emotion = low_specialist.predict(scaler_L2_low.transform(sample_combined))[0]
            else:
                final_emotion = high_specialist.predict(scaler_L2_high.transform(sample_combined))[0]

            fold_preds.append(final_emotion)

            if final_emotion != y_target_test[i]:
                error_log.append({
                    'Fold': fold,
                    'Subject_ID': y_actor_test[i],
                    'True_Emotion': EMOTION_DECODER[y_target_test[i]],
                    'Predicted_Emotion': EMOTION_DECODER[final_emotion],
                    'L1_Routed_To': 'Low' if pred_exc_routes[i] == 0 else 'High',
                    'True_Arousal': 'Low' if y_exc_test[i] == 0 else 'High'
                })

        all_y_true.extend(y_target_test)
        all_pipeline_preds.extend(fold_preds)

    # ==========================================
    # STEP 4: THE GRAND FINALE METRICS
    # ==========================================
    print("\n" + "=" * 50)
    print("INDIVIDUAL MODEL PERFORMANCE (CV Average)")
    print("=" * 50)
    print(f"Gatekeeper (L1) Accuracy:   {accuracy_score(all_exc_true, all_exc_pred) * 100:.2f}%")
    print(f"Low Specialist (L2) Acc:    {accuracy_score(all_low_true, all_low_pred) * 100:.2f}%")
    print(f"High Specialist (L2) Acc:   {accuracy_score(all_high_true, all_high_pred) * 100:.2f}%")

    print("\n" + "=" * 50)
    print("TRUE HIERARCHICAL PIPELINE PERFORMANCE")
    print("=" * 50)

    global_acc = accuracy_score(all_y_true, all_pipeline_preds) * 100
    print(f"GLOBAL PIPELINE ACCURACY: {global_acc:.2f}%\n")

    ordered_classes = [0, 3, 4, 1, 2, 5, 6]
    display_labels = [EMOTION_DECODER[cls] for cls in ordered_classes]

    cm = confusion_matrix(all_y_true, all_pipeline_preds, labels=ordered_classes)

    print(f"{'Emotion':<15} | {'Sensitivity (Recall)':<20} | {'Specificity':<15} | {'F1-Score':<10}")
    print("-" * 68)

    for i, cls in enumerate(ordered_classes):
        emotion_name = EMOTION_DECODER[cls]
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        sensitivity = (TP / (TP + FN)) * 100 if (TP + FN) > 0 else 0
        specificity = (TN / (TN + FP)) * 100 if (TN + FP) > 0 else 0
        precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0
        recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{emotion_name:<15} | {sensitivity:>18.2f}% | {specificity:>13.2f}% | {f1:>8.4f}")

    macro_f1 = f1_score(all_y_true, all_pipeline_preds, average='macro')
    print("-" * 68)
    print(f"MACRO AVERAGE F1 SCORE: {macro_f1:.4f}\n")

    if error_log:
        error_df = pd.DataFrame(error_log)
        csv_path = "pipeline_error_log.csv"
        error_df.to_csv(csv_path, index=False)
        print(f" Saved {len(error_log)} misclassifications to '{csv_path}'. Inspect this file to find actor patterns!")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Purples, ax=ax)
    plt.title("True Pipeline CM (L1 Routing -> L2 Prediction)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("true_pipeline_cm.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
