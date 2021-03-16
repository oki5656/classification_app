"""Models"""

from typing import Dict

from configs.supported_model import SUPPORTED_MODEL

from .resnet import ResNet
#from .vgg import VGG
#from .inception import Inception
#from .mobilenet import MobileNet
#from .resnext import ResNext

def get_model(config: Dict) -> object:
    """Get Model Class"""
    model_name = config['model']['name']

    if model_name in SUPPORTED_MODEL['ResNet']:
        return ResNet(config)
    else:
        NotImplementedError('The model is not supported.')