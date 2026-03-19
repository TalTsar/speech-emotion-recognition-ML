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
Differentiating nuanced emotions (e.g., Sadness vs. Boredom) requires more specialized features. To achieve this, Transfer Learning via Facebook's pre-trained `wav2vec2-base-960h` model was used. Dimentionality reduction using PCA was then used to transform **Wav2Vec2 Embeddings' 768 dimensions** to 2 dimensions, a more appropriate representation for classical ML models.  
  
### Data Storage & Leakage Prevention
All extracted features, alongside their ground-truth emotion labels, are saved as `.npy` (NumPy matrices) to ensure rapid loading during the training phase. 


## Train
`train.py` performs both train and evaluation of the model.
Train-Test split is done using a Subject-Stratified approach.

## Test
<img width="500" height="500" alt="true_pipeline_cm" src="https://github.com/user-attachments/assets/ed9d59b5-0fe5-41ba-a701-b5751f78b669" />
