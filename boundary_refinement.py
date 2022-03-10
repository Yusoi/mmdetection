from ast import Mod
import numpy as np
import scipy 
from multiprocessing import Pool
from enum import Enum

class Direction(Enum):
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7

transform = {
    'N': (-1,0),
    'NE': (-1,1),
    'E': (0,1),
    'SE': (1,1),
    'S': (1,0),
    'SW': (-1,1),
    'W': (-1,0),
    'NW': (-1,-1)
}




"""def _is_border(kernel):
    if kernel[0][1] and kernel[1][0] and kernel[2][1] and kernel[1][2]:
        return True
    else:
        return False"""

# [[0,1,0],
#  [1,1,1],
#  [0,1,0]]

def form_boundary_mask(mask):
    #padded_mask = np.pad(mask, [(1,1),(1,1)], mode='constant', constant_values=[(1,1),(1,1)])
    #padded_mask_shape = padded_mask.shape
    #boundary_mask = np.zeros(mask.shape,dtype=np.bool_)
    kernel = [[0,1,0],[1,1,1],[0,1,0]]
    boundary_mask = scipy.ndimage.convolve(mask,kernel,mode="constant",cval=0.0)
    boundary_mask = (boundary_mask < 5) & (boundary_mask > 0)

    return boundary_mask

def _transform_dir(direction):
    return transform[Direction(direction).name]

def _find_next_dir(last_dir):
    return Direction((Direction[last_dir]-3)%8)

def _find_highest_pixel(boundary):
    y = min(np.where(boundary == 1)[0])
    x = min(np.where(boundary[y] == 1)[0])

    return (x,y)

def order_boundary_pixels(boundary):
    #TODO: Find lowest y and x mask pixel (Highest Pixel)
    #TODO: There might be inside pixels as in the case of a tennis racket and as such this process needs to be verified every time
    
    ordered_pixels = []
    cur_list = []

    cur_pixel = _find_highest_pixel(boundary)
    cur_list.append(cur_pixel)
    next_dir = Direction.E
    


    

    return ordered_pixels

def gen_sliding_window():
    sliding_window = []

    return sliding_window