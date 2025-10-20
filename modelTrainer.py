# To install the below packages, run the following commands in terminal

# pip install ultralytics
# pip install ipython
# pip install torch

# Or run "pip install -r Requirements.txt" in terminal to install all the packages

# saves the model in "runs/modelName(which you provide while training)/weights/best.pt"


# ==========================================
# 🧠 YOLOv8 Training Script
# ==========================================
# Train a YOLOv8 model on your custom dataset
# ==========================================

from ultralytics import YOLO
from IPython.display import Image, display
import torch

def main():
    # -------------------------------
    # 1️⃣ Check GPU availability
    # -------------------------------
    gpu_available = torch.cuda.is_available()
    print(f"✅ GPU Available: {gpu_available}")
    if gpu_available:
        print(f"🖥️ GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ GPU not detected. Training will run on CPU.")

    # -------------------------------
    # 2️⃣ Load a pretrained YOLOv8 model
    # -------------------------------
    # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
    model = YOLO('yolov8n.pt')

    # -------------------------------
    # 3️⃣ Train your model
    # -------------------------------
    results = model.train(
        data='./path/to/your/dataset/data.yaml',  # ⚠️ Update this path
        epochs=50,     # Increase if GPU is available
        imgsz=640,     # Input image size
        batch=16,      # Reduce if using CPU
        name='modelName',  # Name for this training run
        patience=20,   # Early stopping patience
        device=0 if gpu_available else 'cpu',
        save=True,
        plots=True,
        cache=True
    )

    # -------------------------------
    # 4️⃣ Print training results
    # -------------------------------
    print("✅ Training complete!")
    print("📁 Best weights saved at: runs/modelName/weights/best.pt")


if __name__ == "__main__":
    main()
