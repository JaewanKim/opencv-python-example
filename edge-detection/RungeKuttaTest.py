import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import random


class Main:

    def __init__(self):
        self.name = 'gradation3.png'
        img = './Image/' + self.name
        self.original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        self.height = self.original_img.shape[0]
        self.width = self.original_img.shape[1]

        self.result_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        self.gradient_vector_field = [[[0, 0] for col in range(self.width)] for row in range(self.height)]
        self.noise_field = [[0 for col in range(self.width)] for row in range(self.height)]

    def white_noise(self):
        # Create White Noise
        for h in range(0, self.height):
            for w in range(0, self.width):
                self.noise_field[h][w] = random.randrange(0, 256)

    def image_gradient_dw(self):
        gradient_vector_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        for h in range(self.height):
            for w in range(1, self.width - 1):
                gradient_vector_field[h][w][0] = (float(self.original_img[h, w + 1]) - float(self.original_img[h, w - 1])) * 0.5
            gradient_vector_field[h][0][0] = float(self.original_img[h, 1]) - float(self.original_img[h, 0])
            gradient_vector_field[h][self.width - 1][0] = float(self.original_img[h, self.width - 1]) - float(self.original_img[h, self.width - 2])

        for w in range(self.width):
            for h in range(1, self.height - 1):
                gradient_vector_field[h][w][1] = (float(self.original_img[h + 1, w]) - float(self.original_img[h - 1, w])) * 0.5
            gradient_vector_field[0][w][1] = float(self.original_img[1, w]) - float(self.original_img[0, w])
            gradient_vector_field[self.height - 1][w][1] = float(self.original_img[self.height - 1, w]) - float(self.original_img[self.height - 2, w])

        return gradient_vector_field

    def image_gradient(self):
        gradient_vector_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        for h in range(0, self.height):
            for w in range(0, self.width):
                if h == 0:
                    gradient_vector_field[h][w][1] = float(self.original_img[h + 1, w]) - float(self.original_img[h, w])
                elif w == 0:
                    gradient_vector_field[h][w][0] = float(self.original_img[h, w + 1]) - float(self.original_img[h, w])
                elif h == (self.height - 1):
                    gradient_vector_field[h][w][1] = float(self.original_img[h, w]) - float(self.original_img[h - 1, w])
                elif w == (self.width - 1):
                    gradient_vector_field[h][w][0] = float(self.original_img[h, w]) - float(self.original_img[h, w - 1])
                else:
                    gradient_vector_field[h][w][0] = (float(self.original_img[h, w + 1]) - float(self.original_img[h, w - 1])) * 0.5
                    gradient_vector_field[h][w][1] = (float(self.original_img[h + 1, w]) - float(self.original_img[h - 1, w])) * 0.5

        return gradient_vector_field

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
                        gx += np.asarray(sobel_gx_local_matrix)[i][j]
                        gy += np.asarray(sobel_gy_local_matrix)[i][j]
                gx /= 9
                gy /= 9

                mg = math.sqrt(gx**2 + gy**2)
                if mg != 0:
                    sobel_gradient_vector_field[h][w] = [gx / mg, gy / mg]

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

    @staticmethod
    def gaussian_weight(length, sigma=None):
        length = int(length)
        if sigma is None:
            sigma = 1.0
        w = np.mgrid[-length:length + 1]
        g = np.exp(-(w ** 2) / (2 * (sigma ** 2))) / (sigma ** 2 * (2 * math.pi))
        return g / g.sum()

    def lic(self, length, ds):

        for h in range(0, int(self.height)):
            for w in range(0, int(self.width)):

                # array C = compute_integral_curve(p)
                vector = self.gradient_vector_field[h][w]

                x = float(w)
                y = float(h)
                curve_list = []
                weight_list = []
                curve_list.append([x, y])
                weight_list.append(length)

                for s in range(1, length):
                    x = x + ds * vector[0]
                    y = y + ds * vector[1]
                    if 0 <= int(x+0.5) < self.width and 0 <= int(y+0.5) < self.height:
                        curve_list.append([x, y])
                        weight_list.append(length-s)
                        vector = self.gradient_vector_field[int(y+0.5)][int(x+0.5)]

                vector = self.gradient_vector_field[h][w]

                for s in range(-length, 0):
                    x = x - ds * vector[0]
                    y = y - ds * vector[1]
                    if 0 <= int(x+0.5) < self.width and 0 <= int(y+0.5) < self.height:
                        curve_list.append([x, y])
                        weight_list.append(length + s)
                        vector = self.gradient_vector_field[int(y+0.5)][int(x+0.5)]

                # sum = compute_convolution(image, C)
                tot: int = 0
                tot2: int = 0
                for i in range(0, len(curve_list)):
                    c = curve_list[i]
                    weight = weight_list[i]
                    x = int(c[0] + 0.5)
                    y = int(c[1] + 0.5)
                    tot += (self.noise_field[y][x] * weight)
                    tot2 += weight

                if len(curve_list) != 0:
                    tot /= tot2

                self.result_img.itemset(h, w, int(tot))

    def rk(self, ds, vector):
        ds = float(ds)
        rk_vector = [0.0, 0.0]

        for i in range(0, 2):
            k1 = vector[i]
            k2 = vector[i] + (ds / 2) * k1
            k3 = vector[i] + (ds / 2) * k2
            k4 = vector[i] + ds * k3
            rk_vector[i] = vector[i] + (ds / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return rk_vector

    def sobel_smoothing(self, vector_field):

        smooth_vector_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        smooth_vector_field = np.asarray(smooth_vector_field)

        for h in range(1, self.height - 1):
            for w in range(1, self.width - 1):

                avg_x = 0
                avg_y = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        avg_x += vector_field[h + i][w + j][0]
                        avg_y += vector_field[h + i][w + j][1]
                avg_x /= 9
                avg_y /= 9
                smooth_vector_field[h][w] = [avg_x, avg_y]

        return smooth_vector_field

    def lic_gaussain(self, length, ds):
        weight_list = self.gaussian_weight(length, 5)

        for h in range(0, int(self.height)):
            for w in range(0, int(self.width)):

                # array C = compute_integral_curve(p)
                vector = self.gradient_vector_field[h][w]

                x = float(w)
                y = float(h)
                curve_list = []
                curve_list.append([x, y])

                for s in range(1, length):
                    vector = self.rk(ds, vector)
                    x = x + ds * vector[0]
                    y = y + ds * vector[1]
                    if 0 <= int(x+0.5) < self.width and 0 <= int(y+0.5) < self.height:
                        curve_list.append([x, y])
                        vector = self.gradient_vector_field[int(y+0.5)][int(x+0.5)]

                vector = self.gradient_vector_field[h][w]

                for s in range(-length, 0):
                    vector = self.rk(ds, vector)
                    x = x - ds * vector[0]
                    y = y - ds * vector[1]
                    if 0 <= int(x+0.5) < self.width and 0 <= int(y+0.5) < self.height:
                        curve_list.append([x, y])
                        vector = self.gradient_vector_field[int(y+0.5)][int(x+0.5)]

                # sum = compute_convolution(image, C)
                tot: int = 0
                tot2: int = 0
                for i in range(0, len(curve_list)):
                    c = curve_list[i]
                    weight = weight_list[i]
                    x = int(c[0] + 0.5)
                    y = int(c[1] + 0.5)
                    tot += (self.noise_field[y][x] * weight)
                    tot2 += weight

                if len(curve_list) != 0:
                    tot /= tot2

                self.result_img.itemset(h, w, int(tot))

    def __main__(self):

        self.white_noise()
        # vector_field = self.image_gradient()
        vector_field = self.image_gradient_sobel()
        smooth_vector_field = self.sobel_smoothing(vector_field)
        self.gradient_vector_field = self.rotate_field(smooth_vector_field, 90.0)

        length = 20
        ds = 1
        self.lic(length, ds)
        # self.lic_gaussain(length, ds)

        file_name = 'lic-rk-smooth-sobel-weightAvg-length' + str(length) + '-ds' + str(ds) + '-' + self.name

        # UI
        plt.subplot()
        plt.imshow(self.result_img, cmap='gray', interpolation='bicubic')
        plt.title(file_name)
        plt.xticks([]), plt.yticks([])
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    Main().__main__()
