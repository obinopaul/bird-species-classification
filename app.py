from ultralytics import YOLO

datapath = r'data'
model = YOLO('yolov8n-cls.pt')
result = model.train(data = datapath,epochs = 500,imgsz = 640)