#!python3
import argparse
import os
import torch
import cv2
import numpy as np
from .experiment import Structure, Experiment
from .concern.config import Configurable, Config
import math


class LionelDB:
    def __init__(self, weights_path, config_path, image_short_side=736, thresh=None, box_thresh=0.6, resize=False, polygon=True):
        self.args = {
            'weights_path': weights_path,
            'image_short_side': image_short_side,
            'thresh': thresh,
            'box_thresh': box_thresh,
            'resize': resize,
            'polygon': polygon,
        }

        conf = Config()
        experiment_args = conf.compile(conf.load(config_path))['Experiment']
        experiment_args.update(cmd=self.args)
        self.experiment = Configurable.construct_class_from_config(experiment_args)
        self.experiment.load('evaluation', **self.args)

        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        model_saver = self.experiment.train.model_saver
        self.structure = self.experiment.structure
        self.model_path = weights_path

        self.model = self.load_model()

    def load_model(self):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)

        return model

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def process(self, image_path):
        all_matircs = {}
        self.model.eval()
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = self.model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
            
        return output  # batch_boxes, batch_scores = output
