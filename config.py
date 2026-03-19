import os

# 1. FILE PATHS
RAW_DATA_PATH = "./emodb_data/"
FEATURE_SAVE_PATH = "./saved_features/"
MODEL_SAVE_PATH = "./saved_models/"

os.makedirs(FEATURE_SAVE_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 2. EMO-DB MAPPINGS
# W: Anger, L: Boredom, E: Disgust, A: Fear, F: Happiness, T: Sadness, N: Neutral
EMOTION_DICT = {
    'W': 'angry', 'L': 'boredom', 'E': 'disgust',
    'A': 'fear', 'F': 'happy', 'T': 'sad', 'N': 'neutral'
}

EMOTION_MAPPING = {
    'W': 0,  # Anger (Wut)
    'L': 1,  # Boredom (Langeweile)
    'E': 2,  # Disgust (Ekel)
    'A': 3,  # Anxiety/Fear (Angst)
    'F': 4,  # Happiness (Freude)
    'T': 5,  # Sadness (Trauer)
    'N': 6   # Neutral
}

EMOTION_DECODER = {
    0: 'Anger', 1: 'Boredom', 2: 'Disgust', 3: 'Anxiety/Fear',
    4: 'Happiness', 5: 'Sadness', 6: 'Neutral'
}

# 3. HIERARCHICAL MAPPINGS
EXCITATION_MAPPING = {
    'L': 0, 'T': 0, 'N': 0,  # Low Excitation (Boredom, Sadness, Neutral)
    'W': 1, 'A': 1, 'F': 1 ,'E': 1,         # High Excitation (Anger, Fear, Happiness)
}

# Neural network targets must start at 0
LOW_EXC_TARGETS = {'L': 0, 'T': 1, 'N': 2} # 3 Classes
HIGH_EXC_TARGETS = {'W': 0, 'A': 1, 'F': 2, 'E': 3}        # 4 Classes

# 4. TRAINING HYPERPARAMETERS
SAMPLE_RATE = 16000
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

# 5. DEBUGGING
TEST_MODE = False
TEST_NUM_FILES = 40