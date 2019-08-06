import cv2
from matplotlib import pyplot as plt


img = './Image/lenna.png'

img2 = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
laplacian = cv2.Laplacian(img2, cv2.CV_8U)
canny = cv2.Canny(img2, 30, 70)

sobelx3 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
sobely3 = cv2.Sobel(img2, cv2.CV_8U, 0, 1, ksize=3)
sobelxy3 = cv2.Sobel(img2, cv2.CV_8U, 1, 1, ksize=3)
sobelx5 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=5)
sobely5 = cv2.Sobel(img2, cv2.CV_8U, 0, 1, ksize=5)
sobelxy5 = cv2.Sobel(img2, cv2.CV_8U, 1, 1, ksize=5)


images = [#img2, laplacian, canny,
          sobelx3, sobely3, sobelxy3,
          sobelx5, sobely5, sobelxy5]
titles = [#'original', 'laplacian', 'canny',
          'cv2.sobel x ksize=3', 'cv2.sobel y ksize=3', 'cv2.sobel xy ksize=3',
          'cv2.sobel x ksize=5', 'cv2.sobel y ksize=5', 'cv2.sobel xy ksize=5']

for i in range(0, 6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], cmap='gray', interpolation='bicubic'), plt.title([titles[i]])
    plt.xticks([]), plt.yticks([])

plt.show()
