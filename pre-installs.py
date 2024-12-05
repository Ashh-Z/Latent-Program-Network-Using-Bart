import subprocess
import os
import requests


print("Installing dependencies from requirements.txt...")
subprocess.check_call(
    [os.sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"]
)
print("Dependencies installed successfully!")


import gdown

data_folder = "arc-prize-2024"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    folder_url = "https://drive.google.com/drive/folders/1ABHl8uJqaQAvf3Yp0sPRYYSqz_WysE08?usp=sharing"

    print(f"Downloading data from {folder_url}...")
    gdown.download_folder(url=folder_url, output=data_folder, quiet=False)
    print("Data downloaded successfully!")

else:
    print(f"The folder '{data_folder}' already exists. Skipping download.")


save_path = "saved_model/checkpoint.pth"

if not os.path.exists(save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory {save_dir} created.")

    file_url = "https://drive.google.com/uc?export=download&id=1jBShXeeZ0sahu0YwpdeTZ7y43sRq9_5q"
    gdown.download(file_url, save_path, quiet=False)
    print(f"Model downloaded successfully to {save_path}")
else:
    print(f"Model already exists at {save_path}. Skipping download.")

print("Data and model checkpoint downloaded successfully!")
print("Setup complete.")
