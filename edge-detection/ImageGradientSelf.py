import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


class Main:

    def __init__(self):

        self.img = './Image/lenna.png'
        self.original_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        self.threshold_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)

        self.sobel_x_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        self.sobel_y_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        self.sobel_xy_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)

        self.laplacian_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        self.laplacian_d_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)

        self.smooth_sobel_x_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        self.smooth_sobel_y_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        self.smooth_sobel_xy_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)

        self.smooth_laplacian_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        self.smooth_laplacian_d_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)

        self.height = self.original_img.shape[0]
        self.width = self.original_img.shape[1]
        self.max_vector_magnitude = 255 * math.sqrt(2)

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

                sobel_gradient_vector_field[h][w] = [gx/4, gy/4]

        return sobel_gradient_vector_field

    def image_gradient_laplacian(self):
        # Image Gradient - Laplacian

        laplacian_gradient_vector_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        laplacian_d_filter = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])     # Diagonal

        for h in range(1, self.height - 1):
            for w in range(1, self.width - 1):

                local_matrix = np.array([
                    [self.original_img[h - 1, w - 1], self.original_img[h, w - 1], self.original_img[h + 1, w - 1]],
                    [self.original_img[h - 1, w], self.original_img[h, w], self.original_img[h + 1, w + 1]],
                    [self.original_img[h - 1, w + 1], self.original_img[h + 1, w + 1], self.original_img[h + 1, w + 1]]
                ])

                g_local_matrix = np.matmul(laplacian_filter, local_matrix)
                gd_local_matrix = np.matmul(laplacian_d_filter, local_matrix)

                g = 0
                gd = 0

                for i in range(0, 3):
                    for j in range(0, 3):
                        g += np.asarray(g_local_matrix)[i][j]
                        gd += np.asarray(gd_local_matrix)[i][j]

                laplacian_gradient_vector_field[h][w] = [g/4, gd/4]

        return laplacian_gradient_vector_field

    def smoothing(self, vectors):

        vector_field_smooth = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        for h in range(1, self.height - 1):
            for w in range(1, self.width - 1):

                gradient_avg_x = 0
                gradient_avg_y = 0

                for i in range(h - 1, h + 1):
                    for j in range(w - 1, w + 1):
                        gradient_avg_x += vectors[h][w][0]
                        gradient_avg_y += vectors[h][w][1]

                gradient_avg_x /= 9
                gradient_avg_y /= 9

                vector_field_smooth[h][w] = [gradient_avg_y, gradient_avg_x]

        return vector_field_smooth

    def __main__(self):
        # Image Gradient
        sobel_image_gradient = self.image_gradient_sobel()
        laplacian_image_gradient = self.image_gradient_laplacian()

        # Smoothing
        smooth_sobel_image_gradient = self.smoothing(sobel_image_gradient)
        smooth_laplacian_image_gradient = self.smoothing(laplacian_image_gradient)

        # Threshold - Non Smoothing
        for h in range(0, self.height):
            for w in range(0, self.width):

                threshold = 110
                # sobel x
                if sobel_image_gradient[h][w][0] > threshold:
                    self.sobel_x_img.itemset(h, w, 255)
                else:
                    self.sobel_x_img.itemset(h, w, 0)

                # sobel y
                if sobel_image_gradient[h][w][1] > threshold:
                    self.sobel_y_img.itemset(h, w, 255)
                else:
                    self.sobel_y_img.itemset(h, w, 0)

                # sobel xy
                magnitude = math.sqrt(math.pow(sobel_image_gradient[h][w][0], 2) + math.pow(sobel_image_gradient[h][w][1], 2))
                threshold = 200
                if magnitude > threshold:
                    self.sobel_xy_img.itemset(h, w, 255)
                else:
                    self.sobel_xy_img.itemset(h, w, 0)

                # laplacian
                threshold = 8
                if laplacian_image_gradient[h][w][0] > threshold:
                    self.laplacian_img.itemset(h, w, 255)
                else:
                    self.laplacian_img.itemset(h, w, 0)

                # laplacian diagonal
                threshold = 19
                if laplacian_image_gradient[h][w][1] > threshold:
                    self.laplacian_d_img.itemset(h, w, 255)
                else:
                    self.laplacian_d_img.itemset(h, w, 0)

        # Threshold - Smoothing
        for h in range(0, self.height):
            for w in range(0, self.width):

                threshold = 55
                # smooth sobel x
                if smooth_sobel_image_gradient[h][w][0] > threshold:
                    self.smooth_sobel_x_img.itemset(h, w, 255)
                else:
                    self.smooth_sobel_x_img.itemset(h, w, 0)

                # smooth sobel y
                if smooth_sobel_image_gradient[h][w][1] > threshold:
                    self.smooth_sobel_y_img.itemset(h, w, 255)
                else:
                    self.smooth_sobel_y_img.itemset(h, w, 0)

                # smooth sobel xy
                magnitude = math.sqrt(math.pow(smooth_sobel_image_gradient[h][w][0], 2)
                                      + math.pow(smooth_sobel_image_gradient[h][w][1], 2))
                threshold = 100
                if magnitude > threshold:
                    self.smooth_sobel_xy_img.itemset(h, w, 255)
                else:
                    self.smooth_sobel_xy_img.itemset(h, w, 0)

                # smooth laplacian
                threshold = 8
                if smooth_laplacian_image_gradient[h][w][0] > threshold:
                    self.smooth_laplacian_img.itemset(h, w, 255)
                else:
                    self.smooth_laplacian_img.itemset(h, w, 0)

                # smooth laplacian diagonal
                threshold = 4
                if smooth_laplacian_image_gradient[h][w][1] > threshold:
                    self.smooth_laplacian_d_img.itemset(h, w, 255)
                else:
                    self.smooth_laplacian_d_img.itemset(h, w, 0)

        # UI
        images = [self.sobel_x_img, self.sobel_y_img, self.sobel_xy_img, self.laplacian_img, self.laplacian_d_img,
                  self.smooth_sobel_x_img, self.smooth_sobel_y_img, self.smooth_sobel_xy_img,
                  self.smooth_laplacian_img, self.smooth_laplacian_d_img]
        titles = ['sobel x 110', 'sobel y 110', 'sobel xy 200', 'laplacian 8', 'laplacian d 19',
                  'smooth sobel x 55', 'smooth sobel y 55', 'smooth sobel xy 100',
                  'smooth laplacian 8', 'smooth laplacian d 4']

        for i in range(0, 10):
            plt.subplot(2, 5, i+1), plt.imshow(images[i], cmap='gray', interpolation='bicubic'), plt.title([titles[i]])
            plt.xticks([]), plt.yticks([])
        plt.savefig('./Output/ImageGradientSelf/sobel-laplacian-img.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    Main().__main__()

