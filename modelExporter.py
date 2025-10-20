# saves the exported model in "runs/modelName(which you provide while training)/weights/best_web_mode"

# ==========================================
# 🌐 YOLOv8 to TensorFlow.js Export Script
# ==========================================
# Convert your trained YOLOv8 model to TFJS format
# ==========================================

from ultralytics import YOLO

def main():
    # -------------------------------
    # 1️⃣ Load trained model
    # -------------------------------
    model_path = 'runs/modelName/weights/best.pt'  # ⚠️ Update with your trained model path
    model = YOLO(model_path)
    print(f"✅ Loaded model from: {model_path}")

    # -------------------------------
    # 2️⃣ Export to TensorFlow.js format
    # -------------------------------
    model.export(
        format='tfjs',
        imgsz=640,   # Must match training size
        int8=False   # Set True for smaller quantized model
    )

    print("✅ Export complete!")
    print("📁 Model saved in: runs/modelName/weights/best_web_model/")


if __name__ == "__main__":
    main()
