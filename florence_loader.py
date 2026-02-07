import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def _plugin_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_local_hf_env():
    """
    强制把HF/Transformers缓存指到插件目录，避免写系统用户目录。
    """
    base = _plugin_dir()
    hf_home = os.path.join(base, "_hf_cache")
    os.makedirs(hf_home, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_home
    os.environ["HF_HUB_CACHE"] = hf_home


class WZF_LoadFlorence2:
    """
    输出 model + processor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 你把模型文件放这里：custom_nodes/wzf_image2prompt/models/florence2-base
                "model_dir": ("STRING", {"default": os.path.join(_plugin_dir(), "models", "florence2-base")}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["auto", "float16", "bfloat16", "float32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("WZF_FLORENCE2_MODEL", "WZF_FLORENCE2_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load"
    CATEGORY = "WZF"

    def load(self, model_dir, device, dtype):
        _ensure_local_hf_env()

        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        torch_dtype = None
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = None

        # 只读本地，不允许自动去网上拉
        processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )

        if device == "cuda":
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        model.eval()
        return (model, processor)
