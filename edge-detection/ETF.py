from typing import List

import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import random


class Main:

    def __init__(self):
        self.name = 'lenna.png'
        img = './Image/' + self.name
        self.original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        self.height = self.original_img.shape[0]
        self.width = self.original_img.shape[1]

        self.result_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        self.grad_mg = [[0.0 for col in range(self.width)] for row in range(self.height)]
        self.sobel_vector_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]
        self.gradient_vector_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        self.refined_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        self.noise_field = [[0 for col in range(self.width)] for row in range(self.height)]

    def white_noise(self):
        # Create White Noise
        for h in range(0, self.height):
            for w in range(0, self.width):
                self.noise_field[h][w] = random.randrange(0, 256)

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
                    gradient_vector_field[h][w][0] = (float(self.original_img[h, w + 1]) - float(
                        self.original_img[h, w - 1])) * 0.5
                    gradient_vector_field[h][w][1] = (float(self.original_img[h + 1, w]) - float(
                        self.original_img[h - 1, w])) * 0.5

        return gradient_vector_field

    def image_gradient_sobel(self):
        # Image Gradient - Sobel
        sobel_gradient_vector_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        sobel_filter_x = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=float)
        sobel_filter_y = np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]], dtype=float)

        for h in range(1, self.height - 1):
            for w in range(1, self.width - 1):

                local_matrix = np.array([
                    [self.original_img[h - 1, w - 1], self.original_img[h, w - 1], self.original_img[h + 1, w - 1]],
                    [self.original_img[h - 1, w], self.original_img[h, w], self.original_img[h + 1, w + 1]],
                    [self.original_img[h - 1, w + 1], self.original_img[h + 1, w + 1], self.original_img[h + 1, w + 1]]
                ], dtype=float)
                # local_matrix = np.array([
                #     [self.norm_img[h - 1, w - 1], self.norm_img[h, w - 1], self.norm_img[h + 1, w - 1]],
                #     [self.norm_img[h - 1, w], self.norm_img[h, w], self.norm_img[h + 1, w + 1]],
                #     [self.norm_img[h - 1, w + 1], self.norm_img[h + 1, w + 1], self.norm_img[h + 1, w + 1]]
                # ])

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

                if gx == 0.0 and gy == 0.0:
                    gx = 1.0

                # mg = math.sqrt(gx ** 2 + gy ** 2)
                # if mg != 0:
                #     sobel_gradient_vector_field[h][w] = [gx / mg, gy / mg]
                sobel_gradient_vector_field[h][w] = [gx, gy]

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
        # field = self.gradient_vector_field.flatten()
        for h in range(0, int(self.height)):
            for w in range(0, int(self.width)):

                # array C = compute_integral_curve(p)
                vector: List[float] = self.gradient_vector_field[h][w]

                x = float(w)
                y = float(h)
                curve_list = []
                weight_list = []
                curve_list.append([x, y])
                weight_list.append(length)

                for s in range(1, length):
                    x = x + ds * vector[0]
                    y = y + ds * vector[1]
                    # print("x:", x, " y:", y, "s:", s, " is float? ", isinstance(x, float))

                    if 0 <= int(x + 0.5) < self.width and 0 <= int(y + 0.5) < self.height:
                        curve_list.append([x, y])
                        weight_list.append(length - s)
                        vector = self.gradient_vector_field[int(y + 0.5)][int(x + 0.5)]

                vector = self.gradient_vector_field[h][w]

                for s in range(-length, 0):
                    x = x - ds * vector[0]
                    y = y - ds * vector[1]
                    if 0 <= int(x + 0.5) < self.width and 0 <= int(y + 0.5) < self.height:
                        curve_list.append([x, y])
                        weight_list.append(length + s)
                        vector = self.gradient_vector_field[int(y + 0.5)][int(x + 0.5)]

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
                    if 0 <= int(x + 0.5) < self.width and 0 <= int(y + 0.5) < self.height:
                        curve_list.append([x, y])
                        vector = self.gradient_vector_field[int(y + 0.5)][int(x + 0.5)]

                vector = self.gradient_vector_field[h][w]

                for s in range(-length, 0):
                    vector = self.rk(ds, vector)
                    x = x - ds * vector[0]
                    y = y - ds * vector[1]
                    if 0 <= int(x + 0.5) < self.width and 0 <= int(y + 0.5) < self.height:
                        curve_list.append([x, y])
                        vector = self.gradient_vector_field[int(y + 0.5)][int(x + 0.5)]

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

    def spatial_weight(self, h, w, r, c, radius):  # Eq(2)
        if math.sqrt((h - r) ** 2 + (w - c) ** 2) < radius:
            return 1
        else:
            return 0

    def magnitude_weight(self, mg_x, mg_y, n=None):  # Eq(3)
        if n is None:
            n = 1
        return 0.5 * (1 + math.tanh(n * (mg_y - mg_x)))

    def direction_weight(self, vector_x, vector_y):  # Eq(4)
        return abs(np.dot(vector_x, vector_y))

    def get_phi(self, vector_x, vector_y):  # Eq(5)
        if np.dot(vector_x, vector_y) > 0:
            return 1
        else:
            return -1

    def normalize_vector(self, vector):
        normalized_vector = [0.0, 0.0]
        mg = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if mg != 0:
            normalized_vector[0] = vector[0]/mg
            normalized_vector[1] = vector[1]/mg
        return normalized_vector

    def mean_normalize(self, field, norm_min, norm_max):
        result_field = [[0.0 for col in range(self.width)] for row in range(self.height)]

        total = sum(field[i][j] for i, j in field)
        avg = total/len(field)

        for i in range(self.height):
            for j in range(self.width):
                result_field[i][j] = (field[i][j] - avg) / norm_max - norm_min

        return result_field

    def init_ETF(self):

        self.grad = self.image_gradient_sobel()
        grad_mg = [[0.0 for col in range(self.width)] for row in range(self.height)]

        for i in range(0, self.height):
            for j in range(0, self.width):
                grad_mg[i][j] = math.sqrt(self.grad[i][j][0] ** 2 + self.grad[i][j][1] ** 2)

        arr = np.asarray(grad_mg, dtype=float)
        arr = arr.astype('float32')
        self.grad_mg = cv2.normalize(arr, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        flow_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        for h in range(1, self.height-1):
            for w in range(1, self.width-1):
                flow_field[h][w] = self.normalize_vector(self.grad[h][w])

        self.flow_field = self.rotate_field(flow_field, 90)

    def refine_ETF(self, ksize, field=None):
        refined_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        if field is None:
            field = self.flow_field

        for h in range(ksize, self.height - ksize):
            for w in range(ksize, self.width - ksize):

                t_cur_x = field[h][w]
                t_new = [0.0, 0.0]
                k = 0.0

                if (h == 5 and w == 97):
                    print()

                # Eq(1)
                for r in range(h - ksize, h + ksize):
                    for c in range(w - ksize, w + ksize):
                        t_cur_y = field[r][c]

                        phi = float(self.get_phi(t_cur_x, t_cur_y))
                        ws = float(self.spatial_weight(h, w, r, c, ksize))
                        wm = float(self.magnitude_weight(self.grad_mg[h][w], self.grad_mg[r][c]))
                        wd = float(self.direction_weight(t_cur_x, t_cur_y))

                        t_new[0] += phi * t_cur_y[0] * ws * wm * wd
                        t_new[1] += phi * t_cur_y[1] * ws * wm * wd
                        k += phi * ws * wm * wd

                ''' t_new를 normalize 한다 ! '''
                # print(type(t_new[0]))
                # print(type(k))
                tx = float(t_new[0])
                ty = float(t_new[1])

                if (k==0):
                    print(h, w)

                tx /= k
                ty /= k
                # refined_field[h][w][0] = t_new[0]
                # refined_field[h][w][1] = t_new[1]
                refined_field[h][w][0] = tx
                refined_field[h][w][1] = ty

        return refined_field

    def __main__(self):
        # JUST 1 ITERATION FOR TEST

        self.white_noise()
        self.init_ETF()
        field = self.refine_ETF(5)
        field2 = self.refine_ETF(5, field)
        field3 = self.refine_ETF(5, field2)
        self.gradient_vector_field = self.refine_ETF(5, field3)

        # self.gradient_vector_field   # refine 두세번째에 여기 대입

        self.lic(30, 1)
        file_name = 'ETF-Test4-' + self.name

        # UI
        plt.subplot()
        plt.imshow(self.result_img, cmap='gray', interpolation='bicubic')
        plt.title(file_name)
        plt.xticks([]), plt.yticks([])
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    Main().__main__()
