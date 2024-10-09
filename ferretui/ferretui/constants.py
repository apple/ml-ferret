CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Added by Ferret
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"
VOCAB_IMAGE_W = 1000
VOCAB_IMAGE_H = 1000

# GROUNDING PROMPTS
GROUNDING_TEMPLATES = [
	'\nProvide the bounding boxes of the mentioned objects.',
 	'\nInclude the coordinates for each mentioned object.',
	'\nLocate the objects with their coordinates.',
	'\nAnswer in [x1, y1, x2, y2] format.',
	'\nMention the objects and their locations using the format [x1, y1, x2, y2].',
	'\nDraw boxes around the mentioned objects.',
	'\nUse boxes to show where each thing is.',
	'\nTell me where the objects are with coordinates.',
	'\nList where each object is with boxes.',
	'\nShow me the regions with boxes.'
]