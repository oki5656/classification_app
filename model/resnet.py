"""ResNet model"""
from typing import Dict

import torch.nn as nn
import torchvision.models as models

from utils.paths import Paths
from utils.logger import setup_logger, get_logger
from dataloader.dataloader import DataLoader
from metric import metric as Metric
from executor.trainer import Trainer

from .base_model import BaseModel
from .common.device import setup_device, data_parallel
from .common.criterion import make_criterion
from .common.optimizer import make_optimizer
from .common.ckpt import load_ckpt

LOG = get_logger(__name__)

class ResNet(BaseModel):
    """ResNet Model Class"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = None
        self.model_name = self.config.model.name
        self.n_classes = self.config.model.n_classes
        self.classes = self.config.data.classes
        self.batch_size = self.config.train.batch_size
        self.n_gpus = self.config.train.n_gpus
        self.resume = self.config.model.resume

        # dataloader
        self.trainloader = None
        self.valloader = None
        self.testloader = None

        self.paths = None

    def load_data(self, eval: bool):
        """Loads and Preprocess data"""
        self._set_logging()
        
        LOG.info(f'\nLoading {self.config.data.dataroot} dataset...')
        # train
        if not eval:
            # train data
            LOG.info(f' Train data...')
            self.train_img_list, self.train_lbl_list = DataLoader().load_data(self.config.data.dataroot.train, self.config.data.labelpath.train)
            self.trainloader = DataLoader().preprocess_data(self.config.data, self.train_img_list, self.train_lbl_list, self.batch_size, 'train')

            # val data
            LOG.info(f' Validation data...')
            self.val_img_list, self.val_lbl_list = DataLoader().load_data(self.config.data.dataroot.val, self.config.data.labelpath.val)
            self.valloader = DataLoader().preprocess_data(self.config.data, self.val_img_list, self.val_lbl_list, self.batch_size, 'eval')
        
        # evaluation
        if eval:
            LOG.info(f' Test data...')
            self.test_img_list, self.test_lbl_list = DataLoader().load_data(self.config.data.dataroot.test, self.config.data.labelpath.test)
            self.testloader = DataLoader().preprocess_data(self.config.data, self.test_img_list, self.test_lbl_list, self.batch_size, 'eval')

    def _set_logging(self):
        """Set logging"""
        self.paths = Paths.make_dirs(self.config.util.logdir)
        setup_logger(str(self.paths.logdir / 'info.log'))

    def build(self):
        """ Builds model """
        LOG.info(f'\n Building {self.model_name.upper()}...')
        pretrained = self.config.model.pretrained
        print("pretrained=",pretrained)

        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=self.n_classes, bias=True)
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=512, out_features=self.n_classes, bias=True)
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.n_classes, bias=True)
        elif self.model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.n_classes, bias=True)
        elif self.model_name == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
            self.model.fc = nn.Linear(in_features=2048, out_features=self.n_classes, bias=True)
        else:
            raise ValueError('This model name is not supported.')

        # Load checkpoint
        if self.resume:
            ckpt = load_ckpt(self.resume)
            self.model.load_state_dict(ckpt['model_state_dict'])

        LOG.info(' Model was successfully build.')

    def _set_training_parameters(self):
        """Sets training parameters"""
        self.epochs = self.config.train.epochs
        self.save_ckpt_interval = self.config.train.save_ckpt_interval

        # CPU or GPU(single, multi)
        self.device = setup_device(self.n_gpus)
        self.model = self.model.to(self.device)
        if self.n_gpus >= 2:
            self.model = data_parallel(self.model)

        # optimizer and criterion
        self.optimizer = make_optimizer(self.model, self.config.train.optimizer)
        self.criterion = make_criterion(self.config.train.criterion)

        # metric
        self.metric = Metric(self.n_classes, self.classes, self.paths.metric_dir)

    def train(self):
        """Compiles and trains the model"""
        LOG.info('\n Training started.')
        self._set_training_parameters()
        
        train_parameters = {
            'device': self.device,
            'model': self.model,
            'dataloaders': (self.trainloader, self.valloader),
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'metrics': self.metric,
            'save_ckpt_interval': self.save_ckpt_interval,
            'ckpt_dir': self.paths.ckpt_dir,
            'summary_dir': self.paths.summary_dir,
        }

        trainer = Trainer(**train_parameters)
        trainer.train()

    def evaluate(self):
        """Predicts resuts for the test dataset"""
        LOG.info('\n Prediction started...')
        self._set_training_parameters()

        eval_parameters = {
            'device': self.device,
            'model': self.model,
            'dataloaders': (self.trainloader, self.testloader),
            'epochs': None,
            'optimizer': self.optimizer,
            'criterion': self.criterion,
            'metrics': self.metric,
            'save_ckpt_interval': None,
            'ckpt_dir': self.paths.ckpt_dir,
            'summary_dir': self.paths.summary_dir,
        }

        trainer = Trainer(**eval_parameters)
        trainer.eval()