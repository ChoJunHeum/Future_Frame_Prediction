import cv2
import torch
from PIL import Image

from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Images
for f in 'zidane.jpg', 'bus.jpg':
    torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
im1 = Image.open('zidane.jpg')  # PIL image
im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
imgs = [im1, im2]  # batch of images
print(im1)
# Inference
results = model(im1, size=640)  # includes NMS

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

    crop_image = im1.crop(new_area)
    save_image(crop_image,f'crop_imgs/tester_{i}_15.png')
    save_image(im1,f'crop_imgs/tester_{i}.png')
    
    print(new_area)