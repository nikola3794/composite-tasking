
DATA_SPLIT_PATH = "ImageSets/Context"

IMG_PATH = "JPEGImages"
COLOURMAPS_PATH = "colourmaps"
SEG_PATH = "semseg/pascal-context"
PARTS_PATH = "human_superpartssymmetry"
EDGES_PATH = "pascal-context/trainval"
NORMALS_PATH = "normals_distill"
SALIENCY_PATH = "sal_distill"

# Define a numerical relation between the name of a task and its code number
# It is important for code to work properly that the IDs go 0,....,N_tasks-1
TASK_TO_ID = {"seg": 0,
              "parts": 1,
              "edges": 2,
              "saliency": 3,
              "normals": 4, }
ID_TO_TASK = {}
for k in TASK_TO_ID:
    ID_TO_TASK[TASK_TO_ID[k]] = k
TASK_LIST = ["seg", "parts", "edges", "saliency", "normals"]

VOC_CATEGORY_NAMES = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
                    ]
VOC_CLS_ID_TO_NAME = {
    0: "background", 
    1: "cat", 
    2: "aeroplane", 
    3: "chair", 
    4: "pottedplant", 
    5: "sheep", 
    6: "bicycle", 
    7: "cow",
    8: "bird",
    9: "diningtable", 
    10: "sofa", 
    11: "train", 
    12: "boat", 
    13: "dog", 
    14: "bottle", 
    15: "horse", 
    16: "tvmonitor", 
    17: "bus",
    18: "motorbike", 
    19: "car",
    20: "person", 
}
VOC_CLS_NAME_TO_ID = {value : key for (key, value) in VOC_CLS_ID_TO_NAME.items()}
for k in VOC_CLS_NAME_TO_ID:
    assert k in VOC_CATEGORY_NAMES

# Divide all the VOC semantic classes into groups with respect to which task is meaningful to do on them
# Rule R2
VOC_GROUPS_2 = {
    "parts": ['person'],
    "seg": ['bird', 'horse', 'cow', 'cat', 'dog', 'sheep'],
    "normals": ['bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', "tvmonitor"], 
    "saliency": ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train'],
    "edges": ["background"]
}
# Rule R3
VOC_GROUPS_3 = {
    "parts": ['person'],
    "seg": [],
    "normals": ['chair', 'diningtable', 'sofa',   'bicycle', 'bus', 'car', 'motorbike', 'train'], 
    "saliency": ['aeroplane', 'boat',    'bird', 'horse', 'cow', 'cat', 'dog', 'sheep',     'bottle', 'pottedplant', "tvmonitor"],
    "edges": ["background"]
}

HUMAN_PART = {'hair': 1, 'head': 1, 'lear': 1, 'lebrow': 1, 'leye': 1, 'lfoot': 6,
              'lhand': 4, 'llarm': 4, 'llleg': 6, 'luarm': 3, 'luleg': 5, 'mouth': 1,
              'neck': 2, 'nose': 1, 'rear': 1, 'rebrow': 1, 'reye': 1, 'rfoot': 6,
              'rhand': 4, 'rlarm': 4, 'rlleg': 6, 'ruarm': 3, 'ruleg': 5, 'torso': 2
              }
HUMAN_PART_CATEGORY_NAMES = {}
for k in HUMAN_PART:
    if HUMAN_PART[k] not in HUMAN_PART_CATEGORY_NAMES:
        HUMAN_PART_CATEGORY_NAMES[HUMAN_PART[k]] = f"{k}, "
    else:
        HUMAN_PART_CATEGORY_NAMES[HUMAN_PART[k]] += f"{k}, "