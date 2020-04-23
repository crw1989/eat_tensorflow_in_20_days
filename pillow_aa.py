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

# ===================目标检测过程中的数据增强方法之：随机裁剪、随机左右翻转、随机平衡
import random

import numpy as np
from PIL import Image, ImageEnhance
from PIL import ImageDraw



def random_left_right_flip(image, boxes):
    '''
    实现image的随机翻转，图片翻转的同时，也完成了对boxes的翻转
    :param image:image转numpy的数组,dtype=np.float32 (width,height,3)
    :param boxes: (N,4)
    :return:
    '''
    if random.random() < 0.5:
        image[...] = image[:, ::-1, :]
        ih, iw = image.shape[0], image.shape[1]
        boxes[:, [0, 2]] = boxes[:, [2, 0]]
        boxes[:, [0, 2]] = iw - 1 - boxes[:, [0, 2]]
    return image, boxes


def random_crop(image, boxes):
    '''
    实现对图片的随机裁剪，裁剪后的图片仍然包含所有的boxes。
    :param image:
    :param boxes:
    :return:
    '''
    assert (np.max(boxes[:, 2]) <= image.shape[1])
    assert (np.max(boxes[:, 3]) <= image.shape[0])
    xy_min = np.min(boxes[:, :2], axis=-2)
    xy_max = np.max(boxes[:, 2:4], axis=-2)

    random_dx_min = int(np.random.uniform(0, xy_min[0]))
    random_dy_min = int(np.random.uniform(0, xy_min[1]))
    random_dx_max = int(np.random.uniform(xy_max[0], image.shape[1] - 1))
    random_dy_max = int(np.random.uniform(xy_max[1], image.shape[0] - 1))

    image = image[random_dy_min:random_dy_max + 1, random_dx_min:random_dx_max + 1, :]
    # 注意：切片索引时，arr[a:b]不包括b
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - random_dx_min
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - random_dy_min
    return image, boxes


def random_shift(image, boxes):
    '''
    实现对图片的随机平移，平移后的图片中仍然保存原始图片中的所有box
    :param image:
    :param boxes:
    :return:
    '''
    ih, iw = image.shape[:2]
    dxdy_min = np.min(boxes[..., :2], axis=-2)
    dxdy_max = image.shape[:2][::-1] - np.max(boxes[..., 2:4], axis=-2) - 1
    new_image = Image.new('RGB', (iw, ih), color=(128, 128, 128))
    random_dx = int(np.random.uniform(-dxdy_min[0], dxdy_max[0]))
    random_dy = int(np.random.uniform(-dxdy_min[1], dxdy_max[1]))
    boxes[:, [0, 2]] = boxes[:, [0, 2]] + random_dx
    boxes[:, [1, 3]] = boxes[:, [1, 3]] + random_dy
    if (random_dx < 0):
        image = image[:, -random_dx:, :]
        random_dx = 0
    if (random_dx > 0):
        image = image[:, :-random_dx]
    if (random_dy < 0):
        image = image[-random_dy:, :, :]
        random_dy = 0
    if (random_dy > 0):
        image = image[:-random_dy, :, :]
    new_image.paste(Image.fromarray(image), (random_dx, random_dy))

    return np.array(new_image), boxes


def resize_to_train_size(image, boxes, train_input_size):
    '''
    实现图片的按比例缩放，同时缩放后的图片粘贴到宽和高为input_size*input_size的中央
    :param image:
    :param boxes:
    :param train_input_size:
    :return:
    '''
    ih, iw = image.shape[:2]
    scale = min(train_input_size / ih, train_input_size / iw)
    new_h, new_w = int(scale * ih), int(scale * iw)
    image = Image.fromarray(image)
    image = image.resize((new_w, new_h))
    new_image = Image.new('RGB', [train_input_size, train_input_size], (128, 128, 128))
    dx, dy = int((train_input_size - new_w) / 2.0), int((train_input_size - new_h) / 2.0)
    new_image.paste(image, (dx, dy))

    boxes[..., [0, 2]] = scale * boxes[..., [0, 2]] + dx
    boxes[..., [1, 3]] = scale * boxes[..., [1, 3]] + dy
    return np.array(new_image), boxes


def draw_image_with_boxes(image, boxes, name):
    '''
    将预测的框，画到图片中实现可识化。
    :param image:
    :param boxes:
    :param name:
    :return:
    '''
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box[:4].astype(np.int32).tolist(), width=2, outline='yellow')
    image.save(name)
