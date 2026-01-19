import os
import zipfile
import urllib.request
import shutil

def main():
    """Download and extract Tiny ImageNet to ./tiny-imagenet-200."""
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "tiny-imagenet-200.zip"
    extract_path = "tiny-imagenet-200"

    # Download if not already present
    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Zip file already exists; skipping download.")

    # Extract if not already extracted
    if not os.path.exists(extract_path):
        print(f"Extracting Tiny ImageNet to {extract_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        # Move extracted folder to exact path if needed
        if not os.path.exists(extract_path) and os.path.exists("tiny-imagenet-200"):
            shutil.move("tiny-imagenet-200", extract_path)
        print("Extraction complete.")
    else:
        print("Dataset already extracted; skipping.")

    val_dir = os.path.join(extract_path, "val")
    val_img_dir = os.path.join(val_dir, "images")
    val_anno_file = os.path.join(val_dir, "val_annotations.txt")

    if os.path.exists(val_img_dir) and os.path.exists(val_anno_file):
        print("Reorganizing validation images into class subfolders...")
        with open(val_anno_file, "r") as f:
            for line in f.readlines():
                img_name, class_id, *_ = line.strip().split()
                class_folder = os.path.join(val_dir, class_id)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                src = os.path.join(val_img_dir, img_name)
                dst = os.path.join(class_folder, img_name)
                if os.path.exists(src):
                    shutil.move(src, dst)
        print("Validation folder reorganization complete.")
        
        # ---- REMOVE leftover images folder ----
        print("Removing leftover 'images' folder...")
        if os.path.exists(val_img_dir):
            shutil.rmtree(val_img_dir)
            print("'images' folder removed.")
        else:
            print("'images' folder already removed.")
    else:
        print("Validation folder already organized or missing files.")

if __name__ == "__main__":
    main()
