import os
import sys
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import cv2

from lib.config import Config
from lib.experiment import Experiment

def draw_annotation(pred=None, img=None):
    # _, label, _ = self.__getitem__(idx)
    # label = self.label_to_lanes(label)
    img_w = 640
    img_h = 360
    img = cv2.resize(img, (img_w, img_h))
    # img_h, _, _ = img.shape
    # Pad image to visualize extrapolated predictions
    pad = 0
    if pad > 0:
        img_pad = np.zeros((img_h + 2 * pad, img_w + 2 * pad, 3), dtype=np.uint8)
        img_pad[pad:-pad, pad:-pad, :] = img
        img = img_pad
    for i, l in enumerate(pred):
        color = (0, 255, 0)
        points = l.points
        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]
        points = points.round().astype(int)
        points += pad
        for curr_p, next_p in zip(points[:-1], points[1:]):
            img = cv2.line(img,
                            tuple(curr_p),
                            tuple(next_p),
                            color=color,
                            thickness=3)
    # data = [(None, None, label)]
    # if pred is not None:
    #     # print(len(pred), 'preds')
    #     fp, fn, matches, accs = self.dataset.get_metrics(pred, idx)
    #     # print('fp: {} | fn: {}'.format(fp, fn))
    #     # print(len(matches), 'matches')
    #     # print(matches, accs)
    #     assert len(matches) == len(pred)
    #     data.append((matches, accs, pred))
    # else:
    #     fp = fn = None

    # for matches, accs, datum in data:
    #     for i, l in enumerate(datum):
    #         if matches is None:
    #             color = GT_COLOR
    #         elif matches[i]:
    #             color = PRED_HIT_COLOR
    #         else:
    #             color = PRED_MISS_COLOR
    #         points = l.points
    #         points[:, 0] *= img.shape[1]
    #         points[:, 1] *= img.shape[0]
    #         points = points.round().astype(int)
    #         points += pad
    #         xs, ys = points[:, 0], points[:, 1]
    #         for curr_p, next_p in zip(points[:-1], points[1:]):
    #             img = cv2.line(img,
    #                             tuple(curr_p),
    #                             tuple(next_p),
    #                             color=color,
    #                             thickness=3 if matches is None else 3)
            # if 'start_x' in l.metadata:
            #     start_x = l.metadata['start_x'] * img.shape[1]
            #     start_y = l.metadata['start_y'] * img.shape[0]
            #     cv2.circle(img, (int(start_x + pad), int(img_h - 1 - start_y + pad)),
            #                radius=5,
            #                color=(0, 0, 255),
            #                thickness=-1)
            # if len(xs) == 0:
            #     print("Empty pred")
            # if len(xs) > 0 and accs is not None:
            #     cv2.putText(img,
            #                 '{:.0f} ({})'.format(accs[i] * 100, i),
            #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad)),
            #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
            #                 fontScale=0.7,
            #                 color=color)
            #     cv2.putText(img,
            #                 '{:.0f}'.format(l.metadata['conf'] * 100),
            #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad - 50)),
            #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
            #                 fontScale=0.7,
            #                 color=(255, 0, 255))
    return img

cfg_path = '/home/LaneATT/experiments/laneatt_r34_tusimple/config.yaml'
cfg = Config(cfg_path)
experiment = Experiment(cfg, mode="Test")

model_path = '/home/LaneATT/experiments/laneatt_r34_tusimple/models/model_0100.pt'
device = torch.device("cuda")
model = cfg.get_model()
model = experiment.load_pretrained_weights(model, model_path)
model = model.to(device)
model.eval()
test_parameters = cfg.get_test_parameters()

with torch.no_grad():
    image = cv2.imread('/home/LaneATT/20.jpg',cv2.IMREAD_COLOR)
    # print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    image_tensor = image_tensor.to(device)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    obj_and_reg_output = model(image_tensor, **test_parameters)

    prediction = model.decode(obj_and_reg_output, as_lanes=True)

    img = (image_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = draw_annotation(img=img, pred=prediction[0])
    cv2.imshow('pred', img)
    cv2.waitKey(0)