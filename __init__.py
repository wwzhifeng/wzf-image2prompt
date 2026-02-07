from .florence_loader import WZF_LoadFlorence2
from .florence_infer import WZF_ImageToPrompt_Florence2

NODE_CLASS_MAPPINGS = {
    "WZF_LoadFlorence2": WZF_LoadFlorence2,
    "WZF_ImageToPrompt_Florence2": WZF_ImageToPrompt_Florence2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WZF_LoadFlorence2": "WZF Load Florence2",
    "WZF_ImageToPrompt_Florence2": "WZF Image → Prompt (Florence2)",
}
