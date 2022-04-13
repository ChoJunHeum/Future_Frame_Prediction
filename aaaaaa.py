import cv2
import torch
from PIL import Image

from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Images
im1 = Image.open('zidane.jpg')  # PIL image
# im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)

im2 = cv2.imread('bus.jpg')  # OpenCV image (BGR to RGB)
# imgs = [im1, im2]  # batch of images

im2 = torch.Tensor(im2)
print(im2.shape)
# Inference
results = model(im2, size=640)  # includes NMS

print(results)  # im1 predictions (tensor)
results.pandas().xyxy[0]  # im1 predictions (pandas)
areas = results.xyxy[0][:,:4]

print(type(results.xyxy))
for i, area in enumerate(areas):
    area = area.tolist()

    xmin = area[0]
    ymin = area[1]
    xmax = area[2]
    ymax = area[3]
    
    n_x = 2
    n_y = 1.5

    xmin = xmin - (n_x-1)*(xmax-xmin)
    ymin = ymin - (n_y-1)*(ymax-ymin)
    xmax = xmax + (n_x-1)*(xmax-xmin)
    ymax = ymax + (n_y-1)*(ymax-ymin)

    new_area = (xmin, ymin, xmax, ymax)

    crop_image = im2[xmin, ymin, xmax, ymax]
    cv2.imwrite(f'crop_imgs/tester_{i}_15.png',crop_image)
    cv2.imwrite(f'crop_imgs/tester_{i}.png',im2)
    
    print(new_area)