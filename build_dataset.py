import os
import argparse
from PIL import Image

from tqdm import tqdm

SIZE = 256

parser = argparse.ArgumentParser(description = "Preparing UCI Newspaper Segmentation Dataset",
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', default = 'data/dataset_segmentation',
                    help = "Directory contains original data")
parser.add_argument('--output_dir', default = 'data/FCN_dataset',
                    help = "Where to write the new data")

def image_mask_split(filename, image_dir, mask_dir):
    """
    Split the original image and mask data, also save all of the date in output_dir
    as PNG file
    Args:
        filename(str) : original file path
        image_dir(str): a sub folder of output_dir for saving original image
        mask_dir(str): a sub folder of output_dir for saving image mask
    """
    image_name = filename.split('/')[2].split('.')[0]
    image = resize(filename)

    if image_name[-1] == 'm':
        image.save(os.path.join(mask_dir, image_name + '.png'))
    else:
        image.save(os.path.join(image_dir, image_name + '.png'))

def resize(filename, size=SIZE):
    """
    Resize the image contained in `filename`
    """
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    return image

def train_test_split(data_dir, types, rates):
    """
    Split the data into training set and testing set

    Args:
        data_dir(str): directory contains original data
        types(list): might be [`train`, `test`, `val`] or [`train`, `val`] 
        rates(list): the rate for each parts, end with 1, start with 0
    """
    image_path = os.path.join(data_dir, 'image')
    mask_path = os.path.join(data_dir, 'mask')
    filenames = os.listdir(image_path)
    filenames = [os.path.join(image_path, f) for f in filenames]
    files_len = len(filenames)
    split_file_list = []

    for i in range(len(rates)-1):
        split_file_list.append(filenames[int(rates[i] * files_len) : int(rates[i + 1] * files_len)])

    for split, file_list in zip(types, split_file_list):
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)
        else:
            print("Warning: {} data dir {} already exists".format(split, split_path))
        # save image
        split_image_path = os.path.join(split_path, 'image')
        split_mask_path = os.path.join(split_path, 'mask')

        if not os.path.exists(split_image_path):
            os.mkdir(split_image_path)
        else:
            print("Warning: {} image data dir {} already exists".format(split, split_image_path))

        if not os.path.exists(split_mask_path):
            os.mkdir(split_mask_path)
        else:
            print("Warning: {} mask data dir {} already exists".format(split, split_mask_path))

        for f in tqdm(file_list):
            f_name = f.split('/')[-1]
            Image.open(f).save(os.path.join(split_image_path, f_name))

        for f in tqdm(file_list):
            f_name = f.split('/')[-1].split('.')[0]
            f_mask = os.path.join(mask_path, f_name + '_m.png')
            Image.open(f_mask).save(os.path.join(split_mask_path, f_name + '_m.png'))
        

if __name__ == '__main__':
    args = parser.parse_args()

    # check input and output directory
    assert os.path.isdir(args.data_dir), "Can't find the dataset at `{}`".format(args.data_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir `{}` already exits".format(args.output_dir))

    # preprocess all the data
    image_dir = os.path.join(args.output_dir, 'image')
    mask_dir = os.path.join(args.output_dir, 'mask')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    else:
        print("Waring: image dir `{}` already exists".format(image_dir))
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    else:
        print("Waring: mask dir `{}` already exists".format(mask_dir))

    filenames = os.listdir(args.data_dir)
    filenames = [os.path.join(args.data_dir, f) for f in filenames]
    # for filename in tqdm(filenames):
    #    image_mask_split(filename, image_dir, mask_dir)
    print("Split data: Done!")
    
    train_test_split('data/FCN_dataset', ['train', 'val'], [0, 0.75, 1])