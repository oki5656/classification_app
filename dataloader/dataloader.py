"""DataLoader class"""
from typing import List
from pathlib import Path
import pandas as pd
import torch.utils.data as data
from dataloader.utils import make_data_list
from dataloader.dataset import Dataset
from dataloader.transform import DataTransform

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(dataroot: str, labelpath: str):
        """load dataset from path
        Parameters
        ----------
        dataroot : str
            path to the image data directory e.g. './data/images/'
        labelpath : str
            path to the label csv e.g. './data/labels/train.csv'
        Returns
        -------
        Tuple of list
            img_list: e.g. ['./data/images/car1.png', './data/images/dog4.png', ...]
            lbl_list: e.g. [3, 5, ...]
        """
        img_list, lbl_list = make_data_list(dataroot, labelpath)
        return (img_list, lbl_list)

    @staticmethod
    def preprocess_data(data_config: object, img_list: List, lbl_list: List, batch_size: int, mode: str):
        """Preprocess dataset
        Parameters
        ----------
        data_config : object
            data configuration
        img_list : List
            a list of image paths
        lbl_list : List
            a list of labels
        batch_size : int
            batch_size
        mode : str
            'train' or 'eval'
        Returns
        -------
        Object : 
            DataLoader instance
        Raises
        ------
        ValueError
            raise value error if the mode is not 'train' or 'eval'
        """
        # transform
        resize = (data_config.img_size[0], data_config.img_size[1])
        color_mean = tuple(data_config.color_mean)
        color_std = tuple(data_config.color_std)
        transform = DataTransform(resize, color_mean, color_std, mode)

        # dataset
        dataset = Dataset(img_list, lbl_list, transform)

        # dataloader
        if mode == 'train':
            return data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        elif mode == 'eval':
            return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        else:
            raise ValueError('the mode should be train or eval. this mode is not supported')
"""
def make_datapath_list(rootpath, train_data='train.txt', test_data='test.txt', img_extension='.jpg', anno_extension='.png'):
    
    Create list of image and annotation data path
    Parameters
    ----------
    rootpath : str
        path to the data directory
    train_data : str
        text file with train filename
    test_data : str
        text file with test filename
    img_extension : str
        extension of image
    anno_extension : str
        extension of annotation
    Returns
    ----------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
    

    img_dir = Path(rootpath) / 'JPEGImages'
    annot_dir = Path(rootpath) / 'SegmentationClass'

    train_filenames = Path(rootpath) / 'ImageSets' / 'Segmentation' / train_data
    test_filenames = Path(rootpath) / 'ImageSets' / 'Segmentation' / test_data

    # create train img and annot path list
    train_img_list = []
    train_annot_list = []

    for line in open(train_filenames):
        line = line.rstrip('\n')
        img_fname = line + img_extension
        img_path = img_dir / img_fname
        anno_fname = line + anno_extension
        annot_path = annot_dir / anno_fname
        train_img_list.append(str(img_path))
        train_annot_list.append(str(annot_path))

    # create test img and annot path list
    test_img_list = []
    test_annot_list = []

    for line in open(test_filenames):
        line = line.rstrip('\n')
        img_fname = line + img_extension
        img_path = img_dir / img_fname
        anno_fname = line + anno_extension
        annot_path = annot_dir / anno_fname
        test_img_list.append(str(img_path))
        test_annot_list.append(str(annot_path))

    return train_img_list, train_annot_list, test_img_list, test_annot_list
"""