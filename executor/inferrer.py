"""Inferrer"""
from typing import Dict
import PIL

import torch
import torch.nn.functional as F

from utils.config import Config
from utils.load import load_yaml
from model import get_model
from model.common.device import setup_device
from dataloader.transform import DataTransform
import streamlit as st
import cv2

class Inferrer:
    def __init__(self, configfile: str):
        # Config
        config = load_yaml(configfile)
        self.config = Config.from_json(config)

        # Builds model
        self.model = get_model(config)
        #self.model = base_model(config)
        self.model.build()
        self.model_name = self.model.model_name

        # device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.model.to(self.device)
        self.model.model.eval()

        # classes
        self.classes = self.model.classes
                
    def preprocess(self, image: PIL.Image) -> torch.Tensor:
        """Preprocess Image
        PIL.Image to Tensor
        """
        resize = (self.config.data.img_size[0], self.config.data.img_size[1])
        color_mean = tuple(self.config.data.color_mean)
        color_std = tuple(self.config.data.color_std)
        transform = DataTransform(resize, color_mean, color_std, mode='eval')
        image,_ = transform(image,image)
        image = image.unsqueeze(0) # torch.Size([1, 3, img_size[0], img_size[1]])

        return image

    def infer(self, image: PIL.Image = None) -> Dict:
        """Infer an image
        Parameters
        ----------
        image : PIL.Image, optional
            input image, by default None
        Returns
        -------
        dict :
            prediction result label and probability
        """
        shape = image.size

        tensor_image = self.preprocess(image)

        with torch.no_grad():
            tensor_image = tensor_image.to(self.device)
            output = self.model.model(tensor_image)
            output = F.softmax(output, dim=1)
            pred = output.argmax(axis=1)
            
            label = pred.cpu().detach().clone()[0].item()
            prob = output[0].cpu().detach().clone()[label].item()

        return {'label': label, 'prob': prob}