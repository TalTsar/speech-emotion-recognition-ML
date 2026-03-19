# speech-emotion-recognition-ML

**Overview:** develop hierarchical model using simple ML and feature embeddings.

### Dataset: EmoDb
Downloaded from : https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb

The EMODB database is the freely available German emotional database. The database is created by the Institute of Communication Science, Technical University, Berlin, Germany. Ten professional speakers (five males and five females) participated in data recording. The database contains a total of 535 utterances. The EMODB database comprises of seven emotions: 1) anger; 2) boredom; 3) anxiety; 4) happiness; 5) sadness; 6) disgust; and 7) neutral. The data was recorded at a 48-kHz sampling rate and then down-sampled to 16-kHz.

## Feature Extraction Pipeline
`feature_extraction.py` processes the files into two distinct mathematical representations, one for each level of the hierarchy.

### Preprocessing
* Downsampling
* Normalization
* Bandpass Filtering

### 1. Acoustic Features (For the Level 1 SVM)
The Initial grouping aims to reduce the complexity of distinguishing 7 classes simultateously. It uses a lightweight acoustic vector of low-level features (detailed below), and a linear SVM classifier.

**Grouping based on PCA:**

<img width="500" height="500" alt="pca_emotion_clusters" src="https://github.com/user-attachments/assets/8dd0cc3b-d29e-4f16-a691-625132d8359f" />

Features were Min-Max normalized in the subject level.

### 2. Deep Embeddings (For the Level 2 Classifiers)
Differentiating nuanced emotions (e.g., Sadness vs. Boredom) requires more specialized features. To achieve this, Transfer Learning via Facebook's pre-trained `wav2vec2-large-xlsr-53` model for multilingual speech. Dimentionality reduction using PCA was then used to transform Wav2Vec2 Embeddings' 1024 dimensions to 50 dimensions, a more appropriate representation for classical ML models. 
Both Level 2 classifiers were SVM.
  
### Data Storage
All extracted features, alongside their ground-truth emotion labels, are saved as `.npy` (NumPy matrices) to ensure rapid loading during the training phase. 


## Train
`train.py` performs both train and evaluation of the model.
Train-Test split is done using a Subject-Stratified approach with 5-fold cross validation.

## Test

--- Starting True Pipeline 5-Fold Cross-Validation ---
Processing Fold 1...
Processing Fold 2...
Processing Fold 3...
Processing Fold 4...
Processing Fold 5...

==================================================
INDIVIDUAL MODEL PERFORMANCE (CV Average)
==================================================
Level 1 model Accuracy:   96.26%
Level 2- Low Accuracy:    94.59%
Level 2- High Accuracy:   80.19%

==================================================
TRUE HIERARCHICAL PIPELINE PERFORMANCE
==================================================
GLOBAL PIPELINE ACCURACY: 82.62%

Emotion         | Sensitivity (Recall) | Specificity     | F1-Score  
--------------------------------------------------------------------
Anger           |              90.55% |         93.63% |   0.8582
Anxiety/Fear    |              69.57% |         96.14% |   0.7111
Happiness       |              63.38% |         95.69% |   0.6618
Boredom         |              92.59% |         98.68% |   0.9259
Disgust         |              69.57% |         98.98% |   0.7711
Sadness         |              91.94% |         98.73% |   0.9120
Neutral         |              88.61% |         97.37% |   0.8696
--------------------------------------------------------------------
MACRO AVERAGE F1 SCORE: 0.8157

<img width="500" height="500" alt="true_pipeline_cm" src="https://github.com/user-attachments/assets/ed9d59b5-0fe5-41ba-a701-b5751f78b669" />
