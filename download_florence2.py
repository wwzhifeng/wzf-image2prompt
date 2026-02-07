import os
from huggingface_hub import snapshot_download

# 改这里
REPO_ID = "microsoft/Florence-2-base" # 仓库的repo id
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models", "florence2-base")

os.makedirs(LOCAL_DIR, exist_ok=True)

snapshot_download(
    repo_id=REPO_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    # 不用系统cache，仍会写一个元数据cache，这里也强行指定到插件目录
    cache_dir=os.path.join(os.path.dirname(__file__), "_hf_cache"),
    resume_download=True,
)

print("OK:", LOCAL_DIR)
