import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import random

img = './Image/lenna.png'
original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

height = original_img.shape[0]
width = original_img.shape[1]

noise_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

gradient_vector_field = [[[0, 0] for col in range(height)] for row in range(width)]
gradient_smooth_vector_field = [[[0, 0] for col2 in range(height)] for row2 in range(width)]

sobel_filter_x = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])

# Create White Noise
for h in range(0, height):
    for w in range(0, width):
        noise_img[h, w] = random.randrange(0, 255)

# Image Gradient - Sobel
max_vector_magnitude = 255 * math.sqrt(2)
for h in range(1, height-1):
    for w in range(1, width-1):

        # gx = original_img[h, w + 1] - original_img[h, w - 1]
        # gy = original_img[h + 1, w] - original_img[h - 1, w]

        local_matrix = np.array([
            [original_img[h - 1, w - 1], original_img[h, w - 1], original_img[h + 1, w - 1]],
            [original_img[h - 1, w], original_img[h, w], original_img[h + 1, w + 1]],
            [original_img[h - 1, w + 1], original_img[h + 1, w + 1], original_img[h + 1, w + 1]]
        ])

        gx_local_matrix = np.matmul(sobel_filter_x, local_matrix)
        gy_local_matrix = np.matmul(local_matrix, sobel_filter_y)

        gx = np.asarray(gx_local_matrix)[1][1]
        gy = np.asarray(gy_local_matrix)[1][1]

        gradient_vector_field[h][w][0] = gx / max_vector_magnitude
        gradient_vector_field[h][w][1] = gy / max_vector_magnitude

# Smoothing
for h in range(1, height - 1):
    for w in range(1, width - 1):

        gradient_avg_x = 0
        gradient_avg_y = 0

        # Get Average Vector
        for i in range(h - 1, h + 1):
            for j in range(w - 1, w + 1):
                gradient_avg_x += gradient_vector_field[h][w][0]
                gradient_avg_y += gradient_vector_field[h][w][1]

        gradient_avg_x /= 9
        gradient_avg_y /= 9

        gradient_smooth_vector_field[h][w] = [gradient_avg_y, gradient_avg_x]

# UI
X, Y = np.meshgrid(np.arange(0, width), np.arange(0, height))
Xs, Ys = np.meshgrid(np.arange(0, width), np.arange(0, height))
x_shape = X.shape

U = np.zeros(x_shape)
V = np.zeros(x_shape)

for h in range(1, height-1):
    for w in range(1, width-1):
        U[w, h] = gradient_vector_field[h][w][0]
        V[w, h] = gradient_vector_field[h][w][1]
        # U[w, h] = gradient_smooth_vector_field[h][w][0]
        # V[w, h] = gradient_smooth_vector_field[h][w][1]

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V, units='xy', scale=0.1, color='black')
plt.grid()
ax.set_aspect('equal')

plt.xlim(0, width), plt.ylim(0, height)
plt.title('non-smooth', fontsize=10), plt.savefig('non-smooth.png', bbox_inches='tight')
# plt.title('smooth', fontsize=10), plt.savefig('smooth.png', bbox_inches='tight')

plt.show()
