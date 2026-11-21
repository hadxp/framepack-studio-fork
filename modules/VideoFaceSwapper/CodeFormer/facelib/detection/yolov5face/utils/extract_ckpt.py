import torch
import sys
sys.path.insert(0,'./CodeFormer.facelib/detection/yolov5face')
model = torch.load('CodeFormer.facelib/detection/yolov5face/yolov5n-face.pt', map_location='cpu')['model']
torch.save(model.state_dict(),'weights/CodeFormer.facelib/yolov5n-face.pth')