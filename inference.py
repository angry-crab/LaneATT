import os
import sys
import torch
import torch.onnx
import torchvision.transforms as transforms
import random
import numpy as np
import cv2
import onnx
import onnxruntime

from lib.config import Config
from lib.experiment import Experiment


cfg_path = '/home/LaneATT/experiments/laneatt_r34_tusimple/config.yaml'
cfg = Config(cfg_path)
experiment = Experiment(cfg, mode="Test")

model_path = '/home/LaneATT/experiments/laneatt_r34_tusimple/models/model_0100.pt'
device = torch.device("cuda")
model = cfg.get_model()
model = experiment.load_pretrained_weights(model, model_path)
# model = model.to(device)
model.eval()
test_parameters = cfg.get_test_parameters()

test_dataset = cfg.get_dataset('test')

# with torch.no_grad():
    # image = cv2.imread('/home/LaneATT/20.jpg',cv2.IMREAD_COLOR)
    # print(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image, label,_ = test_dataset.__getitem__(1)

# transform = transforms.Compose([transforms.ToTensor()])
# image_tensor = transform(image)'

# image_tensor = image.to(device)
image_tensor = torch.unsqueeze(image, 0)

obj_and_reg_output = model(image_tensor, **test_parameters)

# prediction = model.decode(obj_and_reg_output, as_lanes=True)

torch.onnx.export(model,               # model being run
                image_tensor,                         # model input (or a tuple for multiple inputs)
                "LaneATT.onnx",   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                verbose=True,
                opset_version=11,          # the ONNX version to export the model to
                input_names = ['input'],   # the model's input names
                output_names = ['output']) # the model's output names


    # img = (image_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    # img ,_ ,_ = test_dataset.draw_annotation(idx=1, img=img, label=label, pred=prediction[0], cmp=True)
    # cv2.imshow('pred', img)
    # cv2.waitKey(0)