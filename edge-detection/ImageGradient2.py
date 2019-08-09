import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
import random


class Main:

    def __init__(self):

        self.img = './Image/lenna.png'
        self.original_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        self.white_noise = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)

        self.height = self.original_img.shape[0]
        self.width = self.original_img.shape[1]

        self.gradient_vector_field = [[[0, 0] for col in range(self.height)] for row in range(self.width)]
        self.gradient_vector_field_smooth = [[[0, 0] for col2 in range(self.height)] for row2 in range(self.width)]

        self.sobel_filter_x = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        self.sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])

        for h in range(0, self.height):
            for w in range(0, self.width):
                self.white_noise[h, w] = random.randrange(0, 255)

    def __main__(self):

        # Image Gradient - Sobel
        max_vector_magnitude = 255 * math.sqrt(2)
        for h in range(1, self.height-1):
            for w in range(1, self.width-1):

                # gx = original_img[h, w + 1] - original_img[h, w - 1]
                # gy = original_img[h + 1, w] - original_img[h - 1, w]

                local_matrix = np.array([
                    [self.original_img[h - 1, w - 1], self.original_img[h, w - 1], self.original_img[h + 1, w - 1]],
                    [self.original_img[h - 1, w], self.original_img[h, w], self.original_img[h + 1, w + 1]],
                    [self.original_img[h - 1, w + 1], self.original_img[h + 1, w + 1], self.original_img[h + 1, w + 1]]
                ])

                gx_local_matrix = np.matmul(self.sobel_filter_x, local_matrix)
                gy_local_matrix = np.matmul(local_matrix, self.sobel_filter_y)

                gx = np.asarray(gx_local_matrix)[1][1]
                gy = np.asarray(gy_local_matrix)[1][1]

                self.gradient_vector_field[h][w][0] = gx / max_vector_magnitude
                self.gradient_vector_field[h][w][1] = gy / max_vector_magnitude

        # Smoothing
        for h in range(1, self.height - 1):
            for w in range(1, self.width - 1):

                gradient_avg_x = 0
                gradient_avg_y = 0

                # Get Average Vector
                for i in range(h - 1, h + 1):
                    for j in range(w - 1, w + 1):
                        gradient_avg_x += self.gradient_vector_field[h][w][0]
                        gradient_avg_y += self.gradient_vector_field[h][w][1]

                gradient_avg_x /= 9
                gradient_avg_y /= 9

                self.gradient_vector_field_smooth[h][w] = [gradient_avg_y, gradient_avg_x]

        # UI
        x, y = np.meshgrid(np.arange(0, self.width), np.arange(0, self.height))
        x_shape = x.shape

        u = np.zeros(x_shape)
        v = np.zeros(x_shape)

        for h in range(1, self.height - 1):
            for w in range(1, self.width - 1):
                u[w, h] = self.gradient_vector_field[h][w][0]
                v[w, h] = self.gradient_vector_field[h][w][1]
                # u[w, h] = self.gradient_vector_field_smooth[h][w][0]
                # v[w, h] = self.gradient_vector_field_smooth[h][w][1]

        fig, ax = plt.subplots()
        q = ax.quiver(x, y, u, v, units='xy', scale=0.1, color='black')
        plt.grid()
        ax.set_aspect('equal')

        plt.xlim(0, self.width), plt.ylim(0, self.height)
        plt.title('non-smooth', fontsize=10), plt.savefig('non-smooth.png', bbox_inches='tight')
        # plt.title('smooth', fontsize=10), plt.savefig('smooth.png', bbox_inches='tight')

        plt.show()


if __name__ == '__main__':
    Main().__main__()

