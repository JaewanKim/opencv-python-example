import cv2
import math
import random
import numpy as np
from matplotlib import pyplot as plt


class Main:

    def __init__(self):
        # img = './lenna.png'
        img = './gradation3.png'
        self.original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        self.height = self.original_img.shape[0]
        self.width = self.original_img.shape[1]

        self.result_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        self.gradient_vector_field = [[[0.0, 0.0] for w in range(self.width)] for h in range(self.height)]
        self.noise_field = [[0 for w in range(self.width)] for h in range(self.height)]

    def create_noise_image(self):
        for h in range(0, self.height):
            for w in range(0, self.width):
                self.noise_field[h][w] = random.randrange(0, 256)  # 256

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

    def image_gradient(self):
        for h in range(1, self.height-1):
            for w in range(1, self.width - 1):
                self.gradient_vector_field[h][w][0] = (float(self.original_img[h, w + 1]) - float(self.original_img[h, w - 1]))
                self.gradient_vector_field[h][w][1] = (float(self.original_img[h + 1, w]) - float(self.original_img[h - 1, w]))

        # for h in range(self.height):
        #     for w in range(1, self.width - 1):
        #         self.gradient_vector_field[h][w][0] = (float(self.original_img[h, w + 1]) - float(self.original_img[h, w - 1]))
        #     self.gradient_vector_field[h][0][0] = float(self.original_img[h, 1]) - float(self.original_img[h, 0])
        #     self.gradient_vector_field[h][self.width - 1][0] = float(self.original_img[h, self.width - 1]) - float(self.original_img[h, self.width - 2])
        #
        # for w in range(self.width):
        #     for h in range(1, self.height - 1):
        #         self.gradient_vector_field[h][w][1] = (float(self.original_img[h + 1, w]) - float(self.original_img[h - 1, w]))
        #     self.gradient_vector_field[0][w][1] = float(self.original_img[1, w]) - float(self.original_img[0, w])
        #     self.gradient_vector_field[self.height - 1][w][1] = float(self.original_img[self.height - 1, w]) - float(self.original_img[self.height - 2, w])

    def get_edge_direction_field(self):
        self.image_gradient_sobel()
        for h in range(self.height):
            for w in range(self.width):
                vec = self.gradient_vector_field[h][w]
                temp = vec[0]
                vec[0] = -vec[1]
                vec[1] = temp
                # mag = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                # if mag != 0:
                #     vec[0] /= mag
                #     vec[1] /= mag

    def get_lic_image(self):
        size = 20
        for h in range(0, self.height):
            for w in range(0, self.width):
                point_list = []
                weight_list = []
                px = float(w)
                py = float(h)
                px_int = int(px + 0.5)
                py_int = int(py + 0.5)
                point_list.append([px, py])
                weight_list.append(0)
                for i in range(size):
                    vec = self.gradient_vector_field[py_int][px_int]
                    px += vec[0]
                    py += vec[1]
                    px_int = int(px + 0.5)
                    py_int = int(py + 0.5)
                    if (0 <= px_int < self.width) and (0 <= py_int < self.height):
                        point_list.append([px, py])
                        weight_list.append(i)
                    else:
                        break
                px = float(w)
                py = float(h)
                px_int = int(px + 0.5)
                py_int = int(py + 0.5)
                for i in range(size):
                    vec = self.gradient_vector_field[py_int][px_int]
                    px -= vec[0]
                    py -= vec[1]
                    px_int = int(px + 0.5)
                    py_int = int(py + 0.5)
                    if (0 <= px_int < self.width) and (0 <= py_int < self.height):
                        point_list.append([px, py])
                        weight_list.append(i)
                    else:
                        break
                sum_noise = 0
                sum_weight = 0
                index = 0
                for point in point_list:
                    px_int = int(point[0] + 0.5)
                    py_int = int(point[1] + 0.5)
                    sum_noise += (self.noise_field[py_int][px_int] * weight_list[index])
                    sum_weight += weight_list[index]
                    index += 1
                if sum_weight != 0:
                    self.result_img.itemset(h, w, int(sum_noise / sum_weight + 0.5))
                else:
                    self.result_img.itemset(h, w, 0)

        # UI
        plt.subplot()
        plt.imshow(self.result_img, cmap='gray', interpolation='bicubic')
        plt.title('gradation3 lic length=20')
        plt.xticks([]), plt.yticks([])
        plt.savefig('gradation3-lic-length-20.png', bbox_inches='tight')
        plt.show()

    def __main__(self):
        self.create_noise_image()
        self.get_edge_direction_field()
        self.get_lic_image()


if __name__ == '__main__':
    Main().__main__()
