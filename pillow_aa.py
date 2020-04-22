import numpy_aa as np
from PIL import Image

# ====================open、paste
file_path = r'C:\Users\LenovoPC\Pictures\aa.jpg'
image = Image.open(file_path)
# new_image = Image.new('RGB',(800,800),color=(0,0,0))
# new_image.paste(image)
# new_image.show()
# ====================image与numpy互转
# image= Image.open(file_path)
# image = np.array(image)
# print('dtype:',image.dtype) # uint8
# image = Image.fromarray(image)

# =================== 随机裁剪、随机左右翻转、随机平衡
import random

import numpy as np
from PIL import Image, ImageEnhance
from PIL import ImageDraw


def parse_annotation(line):
    path = line.split()[0]
    boxes = np.array([[float(v) for v in part.split(',')] for part in line.split()[1:]])
    image = np.array(Image.open(path))
    image, boxes = random_left_right_flip(image, boxes)
    image, boxes = random_crop(image, boxes)
    image, boxes = random_shift(image, boxes)
    return image, boxes


def random_left_right_flip(image, boxes):
    if random.random() < 0.5:
        image[...] = image[:, ::-1, :]
        ih, iw = image.shape[0], image.shape[1]
        boxes[:, [0, 2]] = boxes[:, [2, 0]]
        boxes[:, [0, 2]] = iw - boxes[:, [0, 2]]
    return image, boxes


def random_crop(image, boxes):
    xy_min = np.min(boxes[..., :2], axis=-2)
    xy_max = np.max(boxes[..., 2:4], axis=-2)

    random_dx_min = int(np.random.uniform(0, xy_min[0]))
    random_dy_min = int(np.random.uniform(0, xy_min[1]))
    random_dx_max = int(np.random.uniform(xy_max[0], image.shape[1] - 1))
    random_dy_max = int(np.random.uniform(xy_max[1], image.shape[1] - 1))

    image = image[:random_dy_max, :random_dx_max, :]
    image = image[random_dy_min:, random_dx_min:, :]
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - random_dx_min
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - random_dy_min

    return image, boxes


def random_shift(image, boxes):
    ih, iw = image.shape[:2]
    dxdy_min = np.min(boxes[..., :2], axis=-2)
    dxdy_max = image.shape[:2][::-1] - np.max(boxes[..., 2:4], axis=-2)
    new_image = Image.new('RGB', (iw, ih), color=(128, 128, 128))
    random_dx = int(np.random.uniform(-dxdy_min[0], dxdy_max[0]))
    random_dy = int(np.random.uniform(-dxdy_min[1], dxdy_max[1]))
    boxes[:, [0, 2]] = boxes[:, [0, 2]] + random_dx
    boxes[:, [1, 3]] = boxes[:, [1, 3]] + random_dy
    if (random_dx < 0):
        image = image[:, -random_dx:, :]
        random_dx = 0
    if (random_dy < 0):
        image = image[-random_dy:, :, :]
        random_dy = 0
    new_image.paste(Image.fromarray(image), (random_dx, random_dy))

    return np.array(new_image), boxes


def resize_to_train_size(image, boxes, train_input_size):
    scale = min(train_input_size / image.shape[:2])
    new_h, new_w = scale * image.shape[:2]
    image = Image.fromarray(image)
    image = image.resize((new_w, new_h))
    new_image = Image.new('RGB', [train_input_size, train_input_size], (128, 128, 128))
    dx, dy = int((train_input_size - new_w) / 2.0), int((train_input_size - new_h) / 2.0)
    new_image.paste(image, (dx, dy))

    boxes[..., [0, 2]] = scale * boxes[..., [0, 2]] + dx
    boxes[..., [1, 3]] = scale * boxes[..., [1, 3]] + dy
    return new_image, boxes


def draw_image_with_boxes(image, boxes):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box[:4].astype(np.int32).tolist(), width=2, outline='yellow')
    image.save('image.png')

# ======================在image 上画box
def draw_image_with_boxes(image, boxes,name):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box[:4].astype(np.int32).tolist(), width=2, outline='yellow')
    image.save(name)
