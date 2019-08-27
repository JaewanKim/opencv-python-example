import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import random


class Main:

    def __init__(self):
        img = './Image/lenna.png'
        self.original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        self.height = self.original_img.shape[0]
        self.width = self.original_img.shape[1]

        self.noise_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        self.result_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        self.gradient_vector_field = [[[0, 0] for col in range(self.width)] for row in range(self.height)]

        # Create White Noise
        for h in range(0, self.height):
            for w in range(0, self.width):
                self.noise_img[h, w] = random.randrange(0, 255)

    def image_gradient_sobel(self):
        # Image Gradient - Sobel
        sobel_gradient_vector_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        sobel_filter_x = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])

        for h in range(1, self.height - 1):
            for w in range(1, self.width - 1):

                local_matrix = np.array([
                    [self.original_img[h - 1, w - 1], self.original_img[h, w - 1], self.original_img[h + 1, w - 1]],
                    [self.original_img[h - 1, w], self.original_img[h, w], self.original_img[h + 1, w + 1]],
                    [self.original_img[h - 1, w + 1], self.original_img[h + 1, w + 1], self.original_img[h + 1, w + 1]]
                ])

                sobel_gx_local_matrix = np.matmul(sobel_filter_x, local_matrix)
                sobel_gy_local_matrix = np.matmul(local_matrix, sobel_filter_y)
                gx = 0
                gy = 0

                # Convolution
                for i in range(0, 3):
                    for j in range(0, 3):
                        # print(gx, gy)
                        gx += np.asarray(sobel_gx_local_matrix)[i][j]
                        gy += np.asarray(sobel_gy_local_matrix)[i][j]

                sobel_gradient_vector_field[h][w] = [gx/4, gy/4]

        return sobel_gradient_vector_field

    def rotate_field(self, vector_field, theta):
        theta = theta / 180.0 * math.pi

        rotated_field = [[[0, 0] for col in range(self.width)] for row in range(self.height)]

        for h in range(0, self.height):
            for w in range(0, self.width):
                vx = vector_field[h][w][0]
                vy = vector_field[h][w][1]
                rotated_field[h][w][0] = vx * math.cos(theta) - vy * math.sin(theta)
                rotated_field[h][w][1] = vx * math.sin(theta) + vy * math.cos(theta)

        return rotated_field

    def __main__(self):

        vector_field = self.image_gradient_sobel()
        self.gradient_vector_field = self.rotate_field(vector_field, 90.0)
        # self.gradient_vector_field = self.image_gradient_sobel()

        length = 3
        ds = 3
        for h in range(0, int(self.height)):
            for w in range(0, int(self.width)):

                # array C = compute_integral_curve(p)
                vector = self.gradient_vector_field[h][w]

                x = w
                y = h
                curve_list = []
                for s in range(0, length):
                    x = x + int(ds * vector[0])
                    y = y + int(ds * vector[1])
                    if 0 <= x < self.width and 0 <= y < self.height:
                        curve_list.append([x, y])
                        vector = self.gradient_vector_field[y][x]

                vector = self.gradient_vector_field[h][w]
                x = w
                y = h

                for s in range(-length, 0):
                    x = x - int(ds * vector[0])
                    y = y - int(ds * vector[1])
                    if 0 <= x < self.width and 0 <= y < self.height:
                        curve_list.append([x, y])
                        vector = self.gradient_vector_field[y][x]

                # sum = compute_convolution(image, C)
                tot: int = 0
                for c in curve_list:
                    x = c[0]
                    y = c[1]
                    tot += self.noise_img[y][x]

                if len(curve_list) != 0:
                    tot /= len(curve_list)
                self.result_img.itemset(h, w, int(tot))

        # UI
        plt.subplot()
        plt.imshow(self.result_img, cmap='gray', interpolation='bicubic')
        plt.title('lenna lic length3 ds3 r')
        plt.xticks([]), plt.yticks([])
        plt.savefig('lenna-lic-length3-ds3-r.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    Main().__main__()
