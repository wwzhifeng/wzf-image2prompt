from .florence_loader import WZF_LoadFlorence2
from .florence_infer import WZF_ImageToPrompt_Florence2
from .caption_extract import WZF_ExtractCaption


NODE_CLASS_MAPPINGS = {
    "WZF_LoadFlorence2": WZF_LoadFlorence2,
    "WZF_ImageToPrompt_Florence2": WZF_ImageToPrompt_Florence2,
    "WZF_ExtractCaption": WZF_ExtractCaption,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "WZF_LoadFlorence2": "WZF Load Florence2",
    "WZF_ImageToPrompt_Florence2": "WZF Image → Prompt (Florence2)",
    "WZF_ExtractCaption": "WZF Extract Caption",
}

