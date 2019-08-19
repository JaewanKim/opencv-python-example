import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


class Main:

    def __init__(self):

        self.img = './Image/lenna.png'
        self.original_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        self.difference_of_gaussian_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)

        self.height = self.original_img.shape[0]
        self.width = self.original_img.shape[1]

    @staticmethod
    def gaussian_kernel(size, sigma):
        size = int(size)
        sigma = float(sigma)
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2))) / (sigma ** 2 * (2 * math.pi))
        return g / g.sum()

    def difference_of_gaussian(self, ksize, sigma1, sigma2):
        gaussian_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        k = int(ksize)
        sigma1 = float(sigma1)
        sigma2 = float(sigma2)

        # Make the Gaussian by calling the function
        gaussian_kernel1 = self.gaussian_kernel(k, sigma1)
        gaussian_kernel2 = self.gaussian_kernel(k, sigma2)

        for h in range(k, self.height-k):
            for w in range(k, self.width-k):

                local_matrix = [[[0.0, 0.0] for i in range(0, 2 * k + 1)] for j in range(0, 2 * k + 1)]

                for i in range(0, 2*k+1):
                    for j in range(0, 2*k+1):
                        local_matrix[i][j] = self.original_img[i + h - k, j + w - k]

                gaussian_local_matrix1 = np.matmul(gaussian_kernel1, local_matrix)
                gaussian_local_matrix2 = np.matmul(gaussian_kernel2, local_matrix)

                g1 = 0
                g2 = 0

                # Convolution
                for i in range(0, 2*k + 1):
                    for j in range(0, 2*k + 1):
                        g1 += np.asarray(gaussian_local_matrix1)[i][j]
                        g2 += np.asarray(gaussian_local_matrix2)[i][j]

                gaussian_field[h][w] = [g1, g2]

        for h in range(k, self.height-k):
            for w in range(k, self.width-k):
                self.difference_of_gaussian_img.itemset(h, w, gaussian_field[h][w][0] - gaussian_field[h][w][1])

    def __main__(self):

        self.difference_of_gaussian(1, 1.0, 1.5)

        # UI
        images = [self.difference_of_gaussian_img]
        titles = ['DoG']

        for i in range(0, 1):
            plt.subplot(1, 1, i + 1)
            plt.imshow(images[i], cmap='gray', interpolation='bicubic'), plt.title([titles[i]])
            plt.xticks([]), plt.yticks([])
        plt.savefig('./Output/DoGSelf/difference-of-gaussian-img-k1.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    Main().__main__()
