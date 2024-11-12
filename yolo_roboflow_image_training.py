from ultralytics import YOLO
from roboflow import Roboflow

model = YOLO('yolov8n.pt')

rf = Roboflow(api_key="zGjCRPBrA5e9YQJSr5vo")
project = rf.workspace("computer-vision-cuvyr").project("facial-recognition-gcop7")
version = project.version(1)
dataset = version.download("yolov8")

if __name__ == "__main__" :
    model.train(data=f"{dataset.location}/data.yaml", epochs=100)

