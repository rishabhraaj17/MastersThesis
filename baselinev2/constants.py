from enum import Enum


class STEP(Enum):
    SEMI_SUPERVISED = 0
    UNSUPERVISED = 1
    EXTRACTION = 3
    ALGO_VERSION_1 = 4
    DEBUG = 5
    METRICS = 6
    FILTER_FEATURES = 7
    NN_EXTRACTION = 8
    CUSTOM_VIDEO = 9
    MINIMAL = 10
    GENERATE_ANNOTATIONS = 11
    VERIFY_ANNOTATIONS = 12


class ObjectDetectionParameters(Enum):
    BEV_TIGHT = {
        'radius': 60,
        'extra_radius': 0,
        'generic_box_wh': 50,
        'detect_shadows': True
    }
    BEV_RELAXED = {
        'radius': 90,
        'extra_radius': 30,
        'generic_box_wh': 80,
        'detect_shadows': True
    }
    SLANTED = {
        'radius': 80,
        'extra_radius': 20,
        'generic_box_wh': (40, 100),
        'detect_shadows': True
    }
