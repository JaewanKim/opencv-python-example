import cv2
from matplotlib import pyplot as plt
import numpy as np


img = './Image/lenna.png'
original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

height = original_img.shape[0]
width = original_img.shape[1]

gradient_vector_field = [[[0.0, 0.0] for col in range(width)] for row in range(height)]

sobel_filter_x = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])

for h in range(1, height - 1):
    for w in range(1, width - 1):

        local_matrix = np.array([
            [original_img[h - 1, w - 1], original_img[h, w - 1], original_img[h + 1, w - 1]],
            [original_img[h - 1, w], original_img[h, w], original_img[h + 1, w + 1]],
            [original_img[h - 1, w + 1], original_img[h + 1, w + 1], original_img[h + 1, w + 1]]
        ])

        sobel_gx_local_matrix = np.matmul(sobel_filter_x, local_matrix)
        sobel_gy_local_matrix = np.matmul(local_matrix, sobel_filter_y)

        gx = 0
        gy = 0

        # Convolution
        for i in range(0, 3):
            for j in range(0, 3):
                gx += np.asarray(sobel_gx_local_matrix)[i][j]
                gy += np.asarray(sobel_gy_local_matrix)[i][j]

        gradient_vector_field[h][w] = [gx/4, gy/4]

# UI
x, y = np.meshgrid(np.arange(0, height), np.arange(0, width))
x_shape = x.shape

u = np.zeros(x_shape)
v = np.zeros(x_shape)

for h in range(1, height - 1):
    for w in range(1, width - 1):
        u[w, h] = gradient_vector_field[h][w][0]
        v[w, h] = gradient_vector_field[h][w][1]

fig, ax = plt.subplots()
q = ax.quiver(x, y, u, v, units='xy', scale=100, color='black')
plt.grid()
ax.set_aspect('equal')

plt.xlim(0, width), plt.ylim(0, height)
plt.title('vector-field-pyplot', fontsize=10)
plt.savefig('./Output/Library/vector-field-pyplot.png', bbox_inches='tight')

plt.show()
