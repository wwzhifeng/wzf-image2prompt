import torch
import numpy as np
from PIL import Image


def _comfy_image_to_pil(image_tensor):
    """
    ComfyUI IMAGE: torch.Tensor, shape [B,H,W,C], float 0..1
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image must be a torch.Tensor (ComfyUI IMAGE)")

    if image_tensor.dim() == 3:
        img = image_tensor
    else:
        img = image_tensor[0]

    img = img.detach().cpu().clamp(0, 1).numpy()
    img = (img * 255.0).astype(np.uint8)

    # [H,W,C]
    return Image.fromarray(img)


class WZF_ImageToPrompt_Florence2:
    """
    输入 image + model + processor -> 输出 STRING
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("WZF_FLORENCE2_MODEL",),
                "processor": ("WZF_FLORENCE2_PROCESSOR",),
                "task": ([
                    "<CAPTION>",
                    "<DETAILED_CAPTION>",
                    "<MORE_DETAILED_CAPTION>",
                    "<OCR>",
                    "<OD>",
                ], {"default": "<DETAILED_CAPTION>"}),
                "max_new_tokens": ("INT", {"default": 128, "min": 16, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "WZF"

    @torch.inference_mode()
    def run(self, image, model, processor, task, max_new_tokens):
        pil = _comfy_image_to_pil(image)

        # Florence2习惯用 task token 作为文本输入
        inputs = processor(text=task, images=pil, return_tensors="pt")

        # 跟随模型所在设备
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 有些实现带 post_process_generation
        try:
            w, h = pil.size
            processed = processor.post_process_generation(text, task=task, image_size=(w, h))
            # caption类通常会返回 dict 或 str
            if isinstance(processed, dict):
                # 尽量拿caption
                if "caption" in processed:
                    text = processed["caption"]
                else:
                    text = str(processed)
            elif isinstance(processed, str):
                text = processed
            else:
                text = str(processed)
        except Exception:
            pass

        # 你要喂给文生图，做个简单清洗
        text = text.strip().replace("\n", " ")
        return (text,)
