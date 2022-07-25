import pathlib

INPUT_DIR = (
    pathlib.Path(__file__)
    .parent.parent.parent.parent
    .joinpath('input', 'amex-default-prediction')
)

TRAIN_DATA_PATH = INPUT_DIR.joinpath('train_data.csv')
TRAIN_LABELS_PATH = INPUT_DIR.joinpath('train_labels.csv')
TEST_DATA_PATH = INPUT_DIR.joinpath('test_data.csv')
SAMPLE_SUBMISSION_PATH = INPUT_DIR.joinpath('sample_submission.csv')
