import numpy as np
from PIL import Image, ImageDraw
import cv2
import os

videodims = (640,512)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
video = cv2.VideoWriter("test_lwir.mp4",fourcc, 3,videodims)

img_path = os.listdir('/home/urp6/workspace/ssd_for_KAISTPD/save_images_lwir')
img_path.sort()

#draw stuff that goes on every frame here
for i in range(len(img_path)):
    img = Image.open(os.path.join('/home/urp6/workspace/ssd_for_KAISTPD/save_images_lwir',img_path[i]),mode='r')
    imtemp = img.copy()
    # draw frame specific stuff here.
    video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
video.release()