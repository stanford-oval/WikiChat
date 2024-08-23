import argparse
from huggingface_hub import upload_folder

def main(repo_id, folder_path):
    upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="dataset",
        multi_commits=True,
        multi_commits_verbose=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a folder to HuggingFace Hub")
    parser.add_argument("--folder_path", type=str, help="The path to the folder to upload")
    parser.add_argument("--repo_id", type=str, help="The repository ID on HuggingFace Hub")
    
    args = parser.parse_args()
    
    main(args.repo_id, args.folder_path)