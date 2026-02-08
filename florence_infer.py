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

                # ===== Prompt Director =====
                "composition": ([
                    "Full Body｜全身",
                    "Medium Shot｜半身",
                    "Close Up｜特写",
                    "Wide Shot｜远景",
                    "Product Focus｜主体居中",
                ], {"default": "Full Body｜全身"}),

                "lighting": ([
                    "Studio Light｜棚拍光",
                    "Soft Daylight｜柔和日光",
                    "Cinematic｜电影感",
                    "High Contrast｜高反差",
                ], {"default": "Soft Daylight｜柔和日光"}),

                "quality": ([
                    "Standard｜标准",
                    "Commercial｜商业",
                    "Editorial｜杂志",
                    "Luxury｜高端",
                ], {"default": "Commercial｜商业"}),

                "material": ([
                    "Natural｜自然",
                    "Premium｜高级",
                    "Glossy｜光泽",
                    "Matte｜哑光",
                ], {"default": "Premium｜高级"}),

                "detail": ([
                    "Balanced｜均衡",
                    "High｜丰富",
                    "Extreme｜极致",
                ], {"default": "High｜丰富"}),

                "safety": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "WZF"

    @torch.inference_mode()
    def run(
        self,
        image,
        model,
        processor,
        task,
        max_new_tokens,
        composition,
        lighting,
        quality,
        material,
        detail,
        safety,
    ):
        pil = _comfy_image_to_pil(image)

        inputs = processor(text=task, images=pil, return_tensors="pt")

        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        try:
            w, h = pil.size
            processed = processor.post_process_generation(text, task=task, image_size=(w, h))
            if isinstance(processed, dict):
                # Florence2 返回通常是 { "<TASK>": "caption" }
                text = processed.get(task) or next(iter(processed.values()), "")
            elif isinstance(processed, str):
                text = processed
            else:
                text = str(processed)
        except Exception:
            pass


        text = text.strip().replace("\n", " ")

        # ==============================
        # Prompt Director Engine
        # ==============================

        comp_map = {
            "Full Body｜全身": "full body framing, standing pose",
            "Medium Shot｜半身": "medium shot, waist up",
            "Close Up｜特写": "close up portrait, face focus",
            "Wide Shot｜远景": "wide shot, environmental framing",
            "Product Focus｜主体居中": "center composition, product focus",
        }

        lighting_map = {
            "Studio Light｜棚拍光": "studio lighting",
            "Soft Daylight｜柔和日光": "soft daylight",
            "Cinematic｜电影感": "cinematic light",
            "High Contrast｜高反差": "high contrast lighting",
        }

        quality_map = {
            "Standard｜标准": "",
            "Commercial｜商业": "commercial photography",
            "Editorial｜杂志": "editorial grade",
            "Luxury｜高端": "luxury, premium production",
        }

        material_map = {
            "Natural｜自然": "",
            "Premium｜高级": "premium texture",
            "Glossy｜光泽": "glossy material",
            "Matte｜哑光": "matte finish",
        }

        detail_map = {
            "Balanced｜均衡": "",
            "High｜丰富": "high detail",
            "Extreme｜极致": "extreme detail, ultra sharp",
        }

        print(composition, lighting, quality)

        parts = [
            comp_map.get(composition, ""),
            lighting_map.get(lighting, ""),
            quality_map.get(quality, ""),
            text,
            material_map.get(material, ""),
            detail_map.get(detail, ""),
            "refined aesthetics, controlled composition, premium visual order",
        ]

        text = ", ".join([p for p in parts if p])

        if safety:
            text += ", correct anatomy, clean structure, no distortion"

        return (text,)

