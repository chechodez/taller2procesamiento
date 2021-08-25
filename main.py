import cv2
import numpy as np

""" FFT class to properly visualize fft of gray image

"""


class TF:
    def __init__(self, imageBW):
        self.image_gray = imageBW
        self.theta = 0
        self.deltatheta = 0

    def set_theta(self, theta, deltatheta):
        self.theta = theta
        self.deltatheta = deltatheta

    def filtering(self):
        image_gray_fft = np.fft.fft2(self.image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # fft visualization
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)

        # pre-computations
        num_rows, num_cols = (self.image_gray.shape[0], self.image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2   # here we assume num_rows = num_columns

        # low pass filter mask
        maskt = np.zeros_like(self.image_gray)
        freq_cut_off = 0.3  # it should less than 1
        radius_cut_off = int(freq_cut_off * half_size)

        #proceso de cálculo de máscara respecto a theta y delta theta
        if(self.theta==0):
            idx_lp=(self.theta-self.deltatheta<=90 + (180 / np.pi) * np.arctan((row_iter - half_size) / (-col_iter + half_size)))&(self.theta+self.deltatheta>=90 + (180 / np.pi) * np.arctan((row_iter - half_size) / (-col_iter + half_size)))|(180-self.deltatheta<=90 + (180 / np.pi) * np.arctan((row_iter - half_size) / (-col_iter + half_size)))&(180>=90 + (180 / np.pi) * np.arctan((row_iter - half_size) / (-col_iter + half_size)))
        else:
            idx_lp = (self.theta - self.deltatheta <= 90 + (180 / np.pi) * np.arctan(
                (row_iter - half_size) / (-col_iter + half_size))) & (
                                 self.theta + self.deltatheta >= 90 + (180 / np.pi) * np.arctan(
                             (row_iter - half_size) / (-col_iter + half_size)))
        idx_lp[int(half_size),int(half_size)]=1
        maskt[idx_lp] = 1
        maskt[int(half_size),int(half_size)]=1

        # filtering via FFT
        mask = maskt  # can also use high or band pass mask

        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)
        cv2.imshow("Original image", self.image_gray)
        cv2.imshow("Filter frequency response", 255 * mask)
        cv2.imshow("Filtered image", image_filtered)
        cv2.waitKey(0)
        return(image_filtered)

