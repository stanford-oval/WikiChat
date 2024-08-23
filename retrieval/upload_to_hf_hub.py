from huggingface_hub import upload_folder

upload_folder(
    folder_path="/mnt/ephemeral_nvme/",
    repo_id="stanford-oval/wikipedia_10-languages_bge-m3_qdrant_index",
    repo_type="dataset",
    multi_commits=True,
    multi_commits_verbose=True,
)
