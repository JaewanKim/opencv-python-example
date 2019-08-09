import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

img = './Image/lenna.png'

original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

img_gradient_diff_x = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
img_gradient_diff_y = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

img_gradient_sobel_x = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
img_gradient_sobel_y = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
img_gradient_sobel_xy = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

smooth_img_x = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
smooth_img_y = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
smooth_img_xy = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

height = original_img.shape[0]
width = original_img.shape[1]

sobel_filter_x = np.matrix([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
sobel_filter_y = np.matrix([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])

for h in range(1, height - 1):
    for w in range(1, width - 1):

        # Calculate simple brightness differences in the x-axis & y-axis direction
        brightness_diff_x = original_img[h, w + 1] - original_img[h, w - 1]
        brightness_diff_y = original_img[h + 1, w] - original_img[h - 1, w]

        img_gradient_diff_x.itemset(h, w, 0 + brightness_diff_x)
        img_gradient_diff_y.itemset(h, w, 0 + brightness_diff_y)

        # Matrix operation using sobel filter (3 * 3)
        # Get pixel values from the original image for each axis
        local_matrix = np.matrix([
                                [original_img[h - 1, w - 1], original_img[h, w - 1], original_img[h + 1, w - 1]],
                                [original_img[h - 1, w], original_img[h, w], original_img[h + 1, w + 1]],
                                [original_img[h - 1, w + 1], original_img[h + 1, w + 1], original_img[h + 1, w + 1]]])

        # Matrix operation
        sobel_gx_local_matrix = np.matmul(sobel_filter_x, local_matrix)
        sobel_gy_local_matrix = np.matmul(local_matrix, sobel_filter_y)

        # Save Image Gradient by pixel
        for i in range(h - 1, h + 1):
            for j in range(w - 1, w + 1):
                img_gradient_sobel_x.itemset(i, j, 128 + sobel_gx_local_matrix[i - h + 1, j - w + 1])
                img_gradient_sobel_y.itemset(i, j, 128 + sobel_gy_local_matrix[i - h + 1, j - w + 1])
                img_gradient_sobel_xy.itemset(i, j, 128 + math.sqrt(math.pow(sobel_gx_local_matrix[i - h + 1, j - w + 1], 2) + math.pow(sobel_gy_local_matrix[i - h + 1, j - w + 1], 2)))


# Set pixel value greater than threshold
threshold_sobel = 154
for h in range(0, height):
    for w in range(0, width):
        if img_gradient_sobel_x[h, w] > threshold_sobel:
            img_gradient_sobel_x.itemset(h, w, 255)
        else:
            img_gradient_sobel_x.itemset(h, w, 0)

        if img_gradient_sobel_y[h, w] > threshold_sobel:
            img_gradient_sobel_y.itemset(h, w, 255)
        else:
            img_gradient_sobel_y.itemset(h, w, 0)

        if img_gradient_sobel_xy[h, w] > threshold_sobel:
            img_gradient_sobel_xy.itemset(h, w, 255)
        else:
            img_gradient_sobel_xy.itemset(h, w, 0)


# Smoothing
for h in range(1, height - 1):
    for w in range(1, width - 1):

        gradient_avg_x = 0
        gradient_avg_y = 0

        # Get Average Vector
        for i in range(h - 1, h + 1):
            for j in range(w - 1, w + 1):
                gradient_avg_x += img_gradient_sobel_x[i, j]
                gradient_avg_y += img_gradient_sobel_y[i, j]

        gradient_avg_x /= 9
        gradient_avg_y /= 9

        for i in range(h - 1, h + 1):
            for j in range(w - 1, w + 1):
                smooth_img_x.itemset(i, j, 128 + gradient_avg_x)
                smooth_img_y.itemset(i, j, 128 + gradient_avg_y)
                smooth_img_xy.itemset(i, j, 128 + math.sqrt(math.pow(gradient_avg_x, 2) + math.pow(gradient_avg_y, 2)))


threshold_smooth = 154
for h in range(0, height):
    for w in range(0, width):
        #
        if smooth_img_x[h, w] > threshold_smooth:
            smooth_img_x.itemset(h, w, 255)
        else:
            smooth_img_x.itemset(h, w, 0)

        if smooth_img_y[h, w] > threshold_smooth:
            smooth_img_y.itemset(h, w, 255)
        else:
            smooth_img_y.itemset(h, w, 0)

        if smooth_img_xy[h, w] > threshold_smooth:
            smooth_img_xy.itemset(h, w, 255)
        else:
            smooth_img_xy.itemset(h, w, 0)
''''''

# UI
images = [original_img, img_gradient_diff_x, img_gradient_diff_y,
          img_gradient_sobel_x, img_gradient_sobel_y, img_gradient_sobel_xy,
          smooth_img_x, smooth_img_y, smooth_img_xy]
titles = ['original', 'difference x', 'difference y',
          'sobel x (3*3) th=154', 'sobel y (3*3) th=154', 'sobel xy (3*3) th=154',
          'smooth x th=154', 'smooth y th=154', 'smooth xy th=154']

for i in range(0, 6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], cmap='gray', interpolation='bicubic'), plt.title([titles[i]])
    plt.xticks([]), plt.yticks([])
plt.show()
