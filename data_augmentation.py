from utils import read_config
import imgaug.augmenters as iaa
import cv2
import os

# Read config file
config = read_config()

# Define constant variables
DATASET_PATH = config['dataset_path']

# Define augmentation techniques
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flip
    iaa.Sometimes(
        0.7,
        iaa.LinearContrast((0.55, 1.45)),  # change contrast of the image
    ),
    iaa.Sometimes(
        0.3,
        iaa.Affine(),  # rotate images
    ),
    iaa.Sometimes(
        0.55,
        iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255), per_channel=True),  # add noise
    ),
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5)),  # blur images
    ),
    iaa.Sometimes(
        0.5,
        iaa.SaltAndPepper(0.1),  # salt and pepper noise
    )
], random_order=True)

# iterate over directories to augment datas
for directory in ['me', 'not-me']:
    folder = os.path.join(DATASET_PATH, directory)
    folder_list = os.listdir(folder)

    for file in folder_list:
        img_path = os.path.join(folder, file)

        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug_img = seq(images=[img])[0]  # feed images to augment sequence
        # Save images
        cv2.imwrite(f'./dataset/{directory}/{file[:-4]}_aug_{i}.png', aug_img)
