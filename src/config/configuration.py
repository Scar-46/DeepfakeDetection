# ----------------------------- Constants -------------------------------------
# *************** Arguments constants *************************
TUNNING = True
RNG_SEED = 0
N_EPOCHS = 10
PATIENCE = 4

# *************** General constants ***************************
PROJECT_NAME = "DeepFake detection"

# *************** Dataset naming constants ********************
ROOT = '140k-real-and-fake-faces/real_vs_fake/real_vs_fake/real-vs-fake/'

TRAIN_DIR = '140k-real-and-fake-faces/real_vs_fake/real_vs_fake/real-vs-fake/train'
VAL_DIR = '140k-real-and-fake-faces/real_vs_fake/real_vs_fake/real-vs-fake/valid'
TEST_DIR = '140k-real-and-fake-faces/real_vs_fake/real_vs_fake/real-vs-fake/test'

# *************** Model config constants ***********************
EFFICIENTNET_B7 = "google/efficientnet-b7"
EFFICIENTNET_B5 = "google/efficientnet-b5"
MODEL_NAME = EFFICIENTNET_B7

BATCH_SIZE = 32
TARGET_SIZE = 128
LEARNING_RATE = 2e-5

# *************** Evaluator Constants ***********************
THRESHOLD = 0.0
CHECKPOINT = "checkpoints/best-checkpoint.ckpt"

# The activation function and the optimization algorithm should
# be here, but their constructions depends on the model data
# so they cannot be included here