from huggingface_hub import HfApi, login

# 1. Log in to HuggingFace
# This will prompt you for your HuggingFace token.
# (Alternatively, you can pass token="YOUR_TOKEN_HERE" into the login function)
login()

api = HfApi()

# 2. Define your repository name
# Format: "your-hf-username/your-dataset-name"
repo_id = "george22294/SafeQIL-dataset"

# 3. Create the repository (if it doesn't exist yet)
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

# 4. Upload the entire directory
print("Starting upload ...")

api.upload_large_folder(
    folder_path="/home/georgepap/PycharmProjects/SafeQIL/experiments/safety_gymnasium/demonstrations",
    repo_id=repo_id,
    repo_type="dataset",
)

print("Upload complete!")
