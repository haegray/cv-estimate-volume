import cv2 as cv
import numpy as np
import os

files = sorted([f for f in os.listdir('./cube_images') if f.endswith('.png')])
for f in files:
    img = cv.imread(f'./cube_images/{f}')
    size = os.path.getsize(f'./cube_images/{f}')
    channels = img.shape[2] if len(img.shape) == 3 else 1
    print(f'{f}: shape={img.shape}, size={size/1024:.1f}KB, channels={channels}')
