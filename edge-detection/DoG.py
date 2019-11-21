import cv2
from matplotlib import pyplot as plt
import numpy as np
import math


class Main:
    """
        feature/1 - DoG.py cannot detect vertical edge
        TODO: Change kernel instead of square matrix
    """

    def __init__(self):

        # self.img = './Image/lenna2.png'
        # self.img = './Image/circle.jpg'
        self.img = './Image/butterfly.png'
        self.original_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)

        self.height = self.original_img.shape[0]
        self.width = self.original_img.shape[1]

    @staticmethod
    def gaussian_kernel(size, size_y, sigma):
        size = int(size)
        size_y = int(size_y)
        sigma = float(sigma)
        x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
        g = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2))) / (sigma ** 2 * (2 * math.pi))
        result = g/g.sum()
        return result

    def difference_of_gaussian(self, ksize, ksize_y, sigma1, sigma2):

        difference_of_gaussian_img = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        gaussian_field = [[[0.0, 0.0] for col in range(self.width)] for row in range(self.height)]

        k = int(ksize)
        ky = int(ksize_y)
        sigma1 = float(sigma1)
        sigma2 = float(sigma2)

        if k > ky:
            ks = k
        else:
            ks = ky

        # Make the Gaussian by calling the function
        gaussian_kernel1 = self.gaussian_kernel(k, ky, sigma1)
        gaussian_kernel2 = self.gaussian_kernel(k, ky, sigma2)

        for h in range(ks, self.height-ks):
            for w in range(ks, self.width-ks):

                local_matrix = [[[0, 0] for i in range(0, 2 * k + 1)] for j in range(0, 2 * ky + 1)]

                for i in range(0, 2*ky + 1):
                    for j in range(0, 2*k + 1):
                        local_matrix[i][j] = self.original_img[i + h - ky, j + w - k]

                gaussian_local_matrix1 = np.dot(gaussian_kernel1, local_matrix)
                gaussian_local_matrix2 = np.dot(gaussian_kernel2, local_matrix)
                # gaussian_local_matrix1 = np.dot(local_matrix, gaussian_kernel1)
                # gaussian_local_matrix2 = np.dot(local_matrix, gaussian_kernel2)

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
                difference_of_gaussian_img.itemset(h, w, -gaussian_field[h][w][0] + gaussian_field[h][w][1])
        #          여 부분에서 뺀 값 바로 set하지 말고 overflow?되는 거 같은데 threshold 지정해줘볼까?
        return difference_of_gaussian_img

    def __main__(self):

        # difference_of_gaussian1 = self.difference_of_gaussian(2, 2, 1.0, 1.5)
        # difference_of_gaussian2 = self.difference_of_gaussian(2, 2, 1.0, 2.0)
        # difference_of_gaussian3 = self.difference_of_gaussian(4, 1, 1.0, 1.5)
        # difference_of_gaussian4 = self.difference_of_gaussian(4, 1, 1.0, 2.0)
        # difference_of_gaussian5 = self.difference_of_gaussian(1, 4, 1.0, 1.5)
        # difference_of_gaussian6 = self.difference_of_gaussian(1, 4, 1.0, 2.0)
        dog1 = self.difference_of_gaussian(1, 1, 1.0, 1.4)
        dog2= self.difference_of_gaussian(1, 1, 1.0, 1.6)

        # UI
        # images = [difference_of_gaussian1, difference_of_gaussian2, difference_of_gaussian3, difference_of_gaussian4,
        #           difference_of_gaussian5, difference_of_gaussian6]
        # titles = ['k2-2/s1.0-1.5', 'k2-2/s1.0-2.0', 'k4-1/s1.0-1.5', 'k4-1/s1.0-2.0',
        #           'k1-4/s1.0-1.5', 'k1-4/s1.0-2.0']
        images = [dog1, dog2]
        titles = ['k1-1/s1.0-1.4', 'k1-1/s1.0-1.6']
        for i in range(0, 2):
            plt.subplot(1, 2, i + 1)
            plt.imshow(images[i], cmap='gray', interpolation='bicubic'), plt.title([titles[i]])
            plt.xticks([]), plt.yticks([])
        # plt.savefig('./Output/DoGSelf/difference-of-gaussian-butterfly.png', bbox_inches='tight')
        plt.savefig('DoG-reversediff-butterfly.png', bbox_inches='tight')
        plt.show()



if __name__ == '__main__':
    Main().__main__()
