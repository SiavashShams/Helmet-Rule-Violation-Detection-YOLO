from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="/home/ss6928/E6691_assign2/mymodel (1).yaml", epochs=20)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
