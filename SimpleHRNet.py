import os

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from models_.hrnet import HRNet


class SimpleHRNet:
    """
    SimpleHRNet class.

    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the human pose on single images.
    Multi-person support with the YOLOv3 detector is also included (and enabled by default).
    """

    def __init__(self, checkpoint_path, interpolation=cv2.INTER_CUBIC):
        self.c = int(checkpoint_path.split('/')[-1].split('_')[2].strip('w'))
        resolution_strings = checkpoint_path.split('/')[-1].split('_')[3].strip('.pth').split('x')
        self.resolution = (int(resolution_strings[0]), int(resolution_strings[1]))
        self.nof_joints = 17
        self.checkpoint_path = checkpoint_path
        self.interpolation = interpolation
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            self.device = torch.device('cuda')

        self.model = HRNet(c=self.c, nof_joints=self.nof_joints)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        old_res = image.shape
        if self.resolution is not None:
            image = cv2.resize(
                image,
                (self.resolution[1], self.resolution[0]),  # (width, height)
                interpolation=self.interpolation
            )

        images = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
        boxes = np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32)  # [x1, y1, x2, y2]

        if images.shape[0] > 0:
            images = images.to(self.device)

            with torch.no_grad():
                out = self.model(images)

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # For each human, for each joint: y, x, confidence
            for i, human in enumerate(out):
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]
        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        return pts[0]
