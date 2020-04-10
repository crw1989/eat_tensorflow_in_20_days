from PIL import Image

file_path = r'C:\Users\LenovoPC\Pictures\aa.jpg'
image = Image.open(file_path)
new_image = Image.new('RGB',(800,800),color=(0,0,0))
new_image.paste(image)
new_image.show()
