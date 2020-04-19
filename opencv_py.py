import cv2
import numpy_aa as np
# ===================imshow
img = cv2.imread(r'aa.jpg')
cv2.imshow('img',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ===================width、height
print(img.shape)

# ===================平移

rows, cols, _ = img.shape
shifted_img = cv2.warpAffine(img,np.array([[1,0,100],[0,1,50]],dtype=np.float32),(2*cols,2*rows))
print(shifted_img.shape)
# cv2.imshow('png',shifted_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ===================旋转
matrix = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
rotated_img  = cv2.warpAffine(img,matrix,(rows*2,cols*2))
cv2.imshow('rotated_img',rotated_img)
cv2.waitKey()
cv2.destroyAllWindows()