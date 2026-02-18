from huggingface_hub import HfApi, HfFolder, upload_file, create_repo

# 1. Log in to Hugging Face (needs to be done manually the first time)
# Run in terminal:
# huggingface-cli login
# Then enter your Access Token (generate one at https://huggingface.co/settings/tokens)

# 2. Set your repository information
repo_id = "zizi-0123/behavior_analysis_dataset"  # e.g.: "jiahejin/afm-training-logs"
file_path = "/home/jjiahe/code/deepresearch_llm_modeling/train/sft/data/webwalkerqa_logs.zip"
path_in_repo = "data/webwalkerqa_logs.zip"   # Path in the repo after upload
private_dataset = True # Set to True to create a private dataset

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", private=private_dataset, exist_ok=True)

# 3. Upload file
upload_file(
    path_or_fileobj=file_path,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="dataset"   # Change to "model" if uploading to a model repo
)

print(f"✅ Successfully uploaded {file_path} to {repo_id}/{path_in_repo}")