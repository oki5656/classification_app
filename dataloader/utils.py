#import pathlib as Path
from pathlib import Path
import os
def make_data_list(rootpath, train_data='train.txt', test_data='test.txt', img_extension='.jpg', anno_extension='.png'):
    """
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
    """
    print("rootpath=",rootpath)
    img_dir = Path(rootpath) / 'img'

    print("isdir=",os.path.isdir(rootpath))
    train_filenames = Path(rootpath) / "data" / "stl10_binary" / "fold_indices.txt"
    test_filenames = Path(rootpath) / "data" / "stl10_binary" / "fold_indices.txt"
    train_img_list = []

    for line in open(train_filenames):#この直後でスライス使う
        line = line.rstrip('\n')#改行コードを外す
        img_fname = line + img_extension
        img_path = img_dir / img_fname
        train_img_list.append(str(img_path))

    # create test img and annot path list
    test_img_list = []
    #test_annot_list = []

    for line in open(test_filenames):#この直後でスライス使う
        line = line.rstrip('\n')
        img_fname = line + img_extension
        img_path = img_dir / img_fname
        test_img_list.append(str(img_path))

    return train_img_list, test_img_list