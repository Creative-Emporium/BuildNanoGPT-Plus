from huggingface_hub import upload_folder

# Define your folder path and repository name
folder_path = "./forupload"
repo_id = "DrNicefellow/nano_llama_1_7b"

# Upload the folder to the repository
upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",  # You can also specify 'dataset' or 'space' if needed
)

print("Folder uploaded successfully!")
