import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

# img = './Image/circle.png'
img = './Image/lenna.png'

original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

img_gradient_diff_x = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
img_gradient_diff_y = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

img_gradient_sobel_x = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
img_gradient_sobel_y = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
img_gradient_sobel_xy = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

height = original_img.shape[0]
width = original_img.shape[1]

sobel_filter = np.matrix([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
# sobel_filter_gy = np.matrix([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
# sobel_filter_gxy = sobel_filter_gx + sobel_filter_gy

for w in range(1, width - 1):
    for h in range(1, height - 1):

        # Calculate simple brightness differences in the x-axis & y-axis direction
        brightness_diff_x = 255 - original_img[h, w + 1] + original_img[h, w - 1]
        brightness_diff_y = 255 - original_img[h + 1, w] + original_img[h - 1, w]

        img_gradient_diff_x.itemset(h, w, brightness_diff_x)
        img_gradient_diff_y.itemset(h, w, brightness_diff_y)

        # Matrix operation using sobel filter (3 * 3)
        # Get pixel values from the original image for each axis
        local_matrix_x = 255 - np.matrix([
                                [original_img[h - 1, w - 1], original_img[h, w - 1], original_img[h + 1, w - 1]],
                                [original_img[h - 1, w], original_img[h, w], original_img[h + 1, w + 1]],
                                [original_img[h - 1, w + 1], original_img[h + 1, w + 1], original_img[h + 1, w + 1]]])

        local_matrix_y = 255 - np.matrix([
                                [original_img[h - 1, w - 1], original_img[h - 1, w], original_img[h - 1, w + 1]],
                                [original_img[h, w - 1], original_img[h, w], original_img[h, w + 1]],
                                [original_img[h + 1, w - 1], original_img[h + 1, w], original_img[h + 1, w + 1]]])

        local_matrix = local_matrix_x + local_matrix_y

        # Matrix operation
        sobel_gx_local_matrix = np.dot(sobel_filter, local_matrix_x)
        sobel_gy_local_matrix = np.dot(sobel_filter, local_matrix_y)
        sobel_gxy_local_matrix = np.dot(sobel_filter, local_matrix)

        # Save Image Gradient by pixel
        for i in range(h - 1, h + 1):
            for j in range(w - 1, w + 1):
                img_gradient_sobel_x.itemset(i, j, sobel_gx_local_matrix[i - h + 1, j - w + 1])
                img_gradient_sobel_y.itemset(i, j, sobel_gy_local_matrix[i - h + 1, j - w + 1])
                img_gradient_sobel_xy.itemset(i, j, sobel_gxy_local_matrix[i - h + 1, j - w + 1])

        # angle = math.tanh(gradient_direction_x / gradient_direction_y)
        # magnitude = math.sqrt(math.pow(gradient_direction_x, 2) + math.pow(gradient_direction_y, 2))

images = [original_img, img_gradient_diff_x, img_gradient_diff_y,
          img_gradient_sobel_x, img_gradient_sobel_y, img_gradient_sobel_xy]
titles = ['original', 'difference x', 'difference y', 'sobel x (3*3)', 'sobel y (3*3)', 'sobel x + y']

for i in range(0, 6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], cmap='gray', interpolation='bicubic'), plt.title([titles[i]])
    plt.xticks([]), plt.yticks([])

plt.show()
