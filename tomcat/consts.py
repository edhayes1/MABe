DIR_DATASETS = "datasets"
AICROWD_CHALLENGE_NAME = "mabe-2022-mouse-triplets"
DEFAULT_FRAME_RATE = 30
DEFAULT_GRID_SIZE=850
NUM_MICE=3
NUM_KEYPOINTS=12
DEFAULT_NUM_TRAINING_POINTS=1600
DEFAULT_NUM_TESTING_POINTS=3736

NOSE = "nose"
EAR_LEFT = "ear_left"
EAR_RIGHT = "ear_right"
NECK = "neck"
FOREPAW_LEFT = "forepaw_left"
FOREPAW_RIGHT = "forepaw_right"
CENTER = "center"
HINDPAW_LEFT = "hindpaw_left"
HINDPAW_RIGHT = "hindpaw_right"
TAIL_BASE = "tail_base"
TAIL_MIDDLE = "tail_middle"
TAIL_TIP = "tail_tip"

STR_BODY_PARTS = [
    NOSE,
    EAR_LEFT,
    EAR_RIGHT,
    NECK,
    FOREPAW_LEFT,
    FOREPAW_RIGHT,
    CENTER,
    HINDPAW_LEFT,
    HINDPAW_RIGHT,
    TAIL_BASE,
    TAIL_MIDDLE,
    TAIL_TIP,
]

NUM_MICE = 3
KEYFRAME_SHAPE = (3, 12, 2)
BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}