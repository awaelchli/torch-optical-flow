from os.path import *

import numpy as np
from PIL import Image

import optical_flow


def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(file_name)
    elif ext == ".bin" or ext == ".raw":
        return np.load(file_name)
    elif ext == ".flo":
        return optical_flow.read(file_name, fmt="middlebury")
    elif ext == ".pfm":
        return optical_flow.read(file_name, fmt="pfm")
    return []
