"""
SynthVision: Dataset Preprocessing & Conditional Label Preparation
"""
import os
import json
import zipfile
from PIL import Image, ImageOps

def preprocess_images(src_dir, dst_dir, target_size=(256, 256)):
    """Pads and resizes images to square resolution using Lanczos interpolation."""
    os.makedirs(dst_dir, exist_ok=True)
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for cls in classes:
        src_cls = os.path.join(src_dir, cls)
        dst_cls = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)

        for file in os.listdir(src_cls):
            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            img = Image.open(os.path.join(src_cls, file)).convert("RGB")
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            padded = ImageOps.pad(img, target_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            
            output_name = os.path.splitext(file)[0] + ".png"
            padded.save(os.path.join(dst_cls, output_name))
            
    print(f"Preprocessed images stored at: {dst_dir}")

def build_conditional_zip(preprocessed_dir, output_zip_path):
    """Packages preprocessed images with conditional dataset.json for StyleGAN2-ADA."""
    classes = sorted([d for d in os.listdir(preprocessed_dir) if os.path.isdir(os.path.join(preprocessed_dir, d))])
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    
    labels = []
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_STORED) as z:
        for folder, class_id in class_map.items():
            folder_path = os.path.join(preprocessed_dir, folder)
            for file in sorted(os.listdir(folder_path)):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    full_path = os.path.join(folder_path, file)
                    arc_path = f"{folder}/{file}"
                    labels.append([arc_path, class_id])
                    z.write(full_path, arc_path)

        z.writestr("dataset.json", json.dumps({"labels": labels}))

    print(f"Generated conditional archive: {output_zip_path} with {len(labels)} labeled samples.")

if __name__ == "__main__":
    src = "./raw_data"
    preprocessed = "./gan_dataset_256"
    zip_target = "./roof_dataset_conditional_256.zip"
    
    if os.path.exists(src):
        preprocess_images(src, preprocessed)
        build_conditional_zip(preprocessed, zip_target)