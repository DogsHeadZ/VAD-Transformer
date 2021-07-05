import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import torchvision.transforms as transforms

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_C4_1x_caffe2.yaml"
# config_file = "../configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)
import cv2

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content))
    # img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]

    return image

def imshow(img, filename):
    unloader = transforms.ToPILImage()
    # img = unloader(img)
    # plt.imshow(img[:, :, [2, 1, 0]])

    plt.imsave(filename, img)
    plt.axis("off")


# from http://cocodataset.org/#explore?id=345434
image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
# image = Image.open
imshow(image, 'image.jpg')

image = cv2.imread('002.jpg')
# print(img.shape)
# unloader = transforms.ToPILImage()
# # img = unloader(img)
image = np.array(image[:,:,[2,1,0]])
# plt.imsave('object.jpg', img)
print('start')
predictions, feats, bbox = coco_demo.run_on_opencv_image(image)
print(feats.shape)
imshow(predictions, 'image4.jpg')