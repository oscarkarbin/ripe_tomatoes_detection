from ultralytics import YOLO


model = YOLO("yolo11s.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="config.yaml", epochs=30, imgsz=640)