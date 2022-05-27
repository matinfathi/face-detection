import numpy as np
import json
import cv2


def read_config(config_dir='./config.json'):
    with open(config_dir, 'r') as f:
        configures = json.load(f)

    return configures


def crop_images(frame, x, y, w, h, k=0.1):
    """
    Crop face images for feeding to the classifier.

    :param frame: whole frame of the image.
    :param x: x coordinates of the face.
    :param y: y coordinates of the face.
    :param w: width of the face.
    :param h: height of the face.
    :param k: ratio for expanding image.
    :return: croped image
    """
    if x - k * w > 0:
        start_x = int(x - k * w)
    else:
        start_x = x
    if y - k * h > 0:
        start_y = int(y - k * h)
    else:
        start_y = y

    end_x = int(x + (1 + k) * w)
    end_y = int(y + (1 + k) * h)

    face_image = frame[start_y:end_y, start_x:end_x]
    face_image = cv2.resize(face_image, (250, 250))
    # shape from (250, 250, 3) to (1, 250, 250, 3)
    face_image = np.expand_dims(face_image, axis=0)

    return face_image
