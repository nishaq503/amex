import pathlib

ROOT_DIR = (
    pathlib.Path(__file__)
    .parent.parent.parent.parent
)

INPUT_DIR = ROOT_DIR.joinpath('input', 'amex-default-prediction')

TRAIN_DATA_PATH = INPUT_DIR.joinpath('train_data.csv')
TRAIN_LABELS_PATH = INPUT_DIR.joinpath('train_labels.csv')
TEST_DATA_PATH = INPUT_DIR.joinpath('test_data.csv')
SAMPLE_SUBMISSION_PATH = INPUT_DIR.joinpath('sample_submission.csv')

WORKING_DIR = ROOT_DIR.joinpath('working', 'amex-default-prediction')
TRAIN_FTR_PATH = WORKING_DIR.joinpath('train_processed.ftr')
TARGET_FTR_PATH = WORKING_DIR.joinpath('target_processed.ftr')
TEST_FTR_PATH = WORKING_DIR.joinpath('test_processed.ftr')
