import numpy_aa as np
from PIL import Image
#====================open、paste
file_path = r'C:\Users\LenovoPC\Pictures\aa.jpg'
image = Image.open(file_path)
# new_image = Image.new('RGB',(800,800),color=(0,0,0))
# new_image.paste(image)
# new_image.show()
#====================image与numpy互转
# image= Image.open(file_path)
# image = np.array(image)
# print('dtype:',image.dtype) # uint8
# image = Image.fromarray(image)
#====================图像左右、上下翻转
img_arr = np.array(image)
# img_top_down_fliped = Image.fromarray(img_arr[::-1,:,:])
# img_top_down_fliped.show()
# image.show()
img_left_right_fliped = Image.fromarray(img_arr[:,::-1,:])
# img_left_right_fliped.show()
#==================== 图像平移
print(image.size)
new_image = Image.new('RGB',(212,303),color=(0,0,0))
new_image.paste(image,(10,10))
print(new_image.size)
