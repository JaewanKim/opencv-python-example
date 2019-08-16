import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import random


img = './Image/butterfly.png'
original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

height = original_img.shape[0]
width = original_img.shape[1]

noise_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

gradient_vector_field = [[[0, 0] for col in range(width)] for row in range(height)]
gradient_smooth_vector_field = [[[0, 0] for col2 in range(width)] for row2 in range(height)]
C = []

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

# result = lic_flow(gradient_smooth_vector_field)
vectors = gradient_smooth_vector_field
vectors = np.asarray(vectors)
len_pix = 10
m, n, two = vectors.shape
if two != 2:
    raise ValueError


empty_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
result = np.zeros((2*len_pix+1, m, n, 2), dtype=np.int16)   # FIXME: int16?
center = len_pix
result[center, :, :, 0] = np.arange(m)[:, np.newaxis]
result[center, :, :, 1] = np.arange(n)[np.newaxis, :]

for i in range(m):
    # print(i, '/', m)
    for j in range(n):
        y = i
        x = j
        fx = 0.5
        fy = 0.5
        for k in range(len_pix):
            vx, vy = vectors[y, x]
            # print(x, y, vx, vy)
            if vx == 0:
                pass
            if vy == 0:
                pass
            if vx > 0:
                tx = (1-fx)/vx
            else:
                tx = -fx/vx
            if vy > 0:
                ty = (1-fy)/vy
            else:
                ty = -fy/vy
            if tx < ty:
                # print("x step")
                if vx > 0:
                    x += 1
                    fy += vy*tx
                    fx = 0.
                else:
                    x -= 1
                    fy += vy*tx
                    fx = 1.
            else:
                # print("y step")
                if vy > 0:
                    y += 1
                    fx += vx*ty
                    fy = 0.
                else:
                    y -= 1
                    fx += vx*ty
                    fy = 1.
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x >= n:
                x = n-1
            if y >= m:
                y = m-1

            # empty_img.itemset(i, j, math.sqrt(math.pow(y, 2) + math.pow(x, 2)))
            result[center+k+1, i, j, :] = y, x
# print(result)
#
for h in range(0, height):
    for w in range(0, width):
        for k in range(0, 21):
            dx = result[k][h][w][0]
            dy = result[k][h][w][1]
            empty_img.itemset(h, w, math.sqrt(math.pow(dy, 2) + math.pow(dx, 2)))

plt.subplot()
plt.imshow(empty_img, cmap='gray', interpolation='bicubic')
plt.title('lic test')
plt.xticks([]), plt.yticks([])
plt.show()

'''
# LIC implemented
empty_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
L = 1
ds = 1
for h in range(0, height):
    print(h)
    for w in range(0, width):

        # array C = compute_integral_curve(p)
        V = gradient_vector_field[h][w]
        x = w
        y = h
        for s in range(0, L):
            x = x + (int)(ds*V[0])
            y = y + (int)(ds*V[1])
            C.append([x, y])
            V = gradient_vector_field[y][x]

        V = gradient_vector_field[h][w]

        for s in range(0, -L):
            x = x - (int)(ds*V[0])
            y = y - (int)(ds*V[1])
            C.append([x, y])
            V = gradient_vector_field[y][x]

        # sum = compute_convolution(image, C)
        tot = 0
        for c in C:
            x = c[0]
            y = c[1]
            tot += original_img[y][x]
        tot = tot / (2 * L + 1)

        # set pixel p on O_img to sum
        empty_img.itemset(h, w, tot)

plt.subplot()
plt.imshow(empty_img, cmap='gray', interpolation='bicubic')
plt.title('lic test')
plt.xticks([]), plt.yticks([])
plt.show()
'''
'''
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
plt.title('non-smooth', fontsize=10), plt.savefig('sobel-non-sobel-smooth.png', bbox_inches='tight')
# plt.title('smooth', fontsize=10), plt.savefig('sobel-smooth.png', bbox_inches='tight')

plt.show()
'''
