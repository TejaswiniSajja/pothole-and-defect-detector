
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
data_yaml = ROOT / "pothole-dataset" / "data.yaml"


def main():
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data_yaml}"
        )

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=30,
        imgsz=416,
        batch=2,
        patience=8,
        workers=0,
        cache=False,
        device="cpu",
        project="runs/pothole",
        name="pothole_img416_fixed",
        plots=True,
        save=True,
        exist_ok=False
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
