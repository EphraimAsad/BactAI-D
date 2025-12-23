# training/hf_sync.py
# ------------------------------------------------------------
# Sync updated data files back to the SAME Hugging Face Space.
# ------------------------------------------------------------

import os
from typing import List, Dict, Any

from huggingface_hub import HfApi, CommitOperationAdd


def push_to_hf(
    paths: List[str],
    commit_message: str = "train: update extended schema, aliases, signals from gold tests",
) -> Dict[str, Any]:

    repo_id = os.getenv("HF_SPACE_REPO_ID")
    token = os.getenv("HF_TOKEN")

    if not repo_id:
        return {
            "ok": False,
            "error": "Missing HF_SPACE_REPO_ID environment variable.",
            "uploaded": [],
        }

    if not token:
        return {
            "ok": False,
            "error": "Missing HF_TOKEN environment variable.",
            "uploaded": [],
        }

    api = HfApi()
    operations = []
    uploaded = []

    for p in paths:
        if not os.path.exists(p):
            continue

        operations.append(
            CommitOperationAdd(path_in_repo=p, path_or_fileobj=p)
        )
        uploaded.append(p)

    if not operations:
        return {
            "ok": False,
            "error": "No existing files to upload.",
            "uploaded": [],
        }

    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        operations=operations,
        commit_message=commit_message,
        token=token,
    )

    return {
        "ok": True,
        "uploaded": uploaded,
        "repo_id": repo_id,
        "commit_message": commit_message,
        "commit_url": commit_info.commit_url,
    }