TARGET_COL = 'Risk'

NUMERICAL_FEATURES = ['Age', 'Credit amount', 'Duration']

CATEGORICAL_FEATURES = [
    'Sex', 'Housing', 'Saving accounts',
    'Checking account', 'Purpose'
]

RANDOM_STATE = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.25   # 0.25 of 0.80 = 0.20 of total

OUTPUT_DIR   = 'outputs/'
PLOTS_DIR    = 'outputs/plots/'
MODELS_DIR   = 'outputs/models/'