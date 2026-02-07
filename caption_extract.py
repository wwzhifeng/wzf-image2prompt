class WZF_ExtractCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "WZF"

    def run(self, text):
        if not isinstance(text, str):
            return (str(text),)

        try:
            start = text.find(":")
            if start != -1:
                result = text[start + 1 :].strip()

                if result.startswith("'") or result.startswith('"'):
                    result = result[1:]

                if result.endswith("'}") or result.endswith('"}'):
                    result = result[:-2]

                return (result.strip(),)
        except Exception:
            pass

        return (text,)
