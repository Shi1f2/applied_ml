from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TASK1_DATA = DATA / "task1"
TASK2_DATA = DATA / "task2"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"

TASK1_TRAIN = TASK1_DATA / "sentiment_analysis_training_data.csv"
TASK1_VAL = TASK1_DATA / "sentiment_analysis_validation_data.csv"
TASK1_TEST = TASK1_DATA / "sentiment_analysis_test_data.csv"

TASK2_TRAIN = TASK2_DATA / "face_alignment_training_data.npz"
TASK2_VAL = TASK2_DATA / "face_alignment_validation_data.npz"
TASK2_TEST = TASK2_DATA / "face_alignment_test_data.npz"
