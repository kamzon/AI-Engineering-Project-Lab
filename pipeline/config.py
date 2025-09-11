from typing import List


class ModelConstants:
    SAM_VARIANT: str = "vit_b"
    SAM_CHECKPOINT_FILENAME: str = "sam_vit_b_01ec64.pth"
    SAM_CHECKPOINT_URL: str = (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )

    IMAGE_MODEL_ID: str = "microsoft/resnet-50"

    IMAGE_LONGEST_SIDE: int = 1024

    DEFAULT_CANDIDATE_LABELS: List[str] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        "building", "road", "sky", "ground", "water"
    ]

    ZERO_SHOT_MODEL_ID: str = "typeform/distilbert-base-uncased-mnli"

    FEW_SHOT_LR: float = 1e-4
    FEW_SHOT_WEIGHT_DECAY: float = 0.0
    FEW_SHOT_MAX_EPOCHS: int = 3
    FEW_SHOT_BATCH_SIZE: int = 8
    FEW_SHOT_FREEZE_BACKBONE: bool = True

    # Directory where a fine-tuned image classifier (and its processor) will be saved.
    # If this directory exists, the pipeline will load the classifier from here instead
    # of the base `IMAGE_MODEL_ID`.
    FINETUNED_MODEL_DIR: str = "pipeline/finetuned/resnet_few_shot"


