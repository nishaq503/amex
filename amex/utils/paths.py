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

if __name__ == '__main__':
    for p in [INPUT_DIR, TRAIN_DATA_PATH, TRAIN_LABELS_PATH, TEST_DATA_PATH, SAMPLE_SUBMISSION_PATH]:
        assert p.exists(), f'Path not found: {p}'
