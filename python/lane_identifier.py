import numpy as np
import matplotlib.pyplot as plt


class LaneIdentifier:
    def __init__(self, smooth_factor):
        self.margin = 100
        self.smooth_factor = smooth_factor
        self.window_width = 50
        self.window_height = 80
        self.recent_centroids = []
        return

    def identify_lanes(self, binary):
        warped_height = binary.shape[0]
        window_centroids = self.find_window_centroids(binary)
        if len(window_centroids) > 0:
            leftx = []
            rightx = []
            for level in range(0, len(window_centroids)):
                leftx.append(window_centroids[level][0])
                rightx.append(window_centroids[level][1])

            res_yvals = np.arange(warped_height - (self.window_height / 2), 0, -self.window_height)
            left_fit = np.polyfit(res_yvals, leftx, 2)
            right_fit = np.polyfit(res_yvals, rightx, 2)

            return True, left_fit, right_fit

        return False, None, None

    def find_window_centroids(self, image):
        window_centroids = []
        window = np.ones(self.window_width)
        image_width = image.shape[1]
        image_height = image.shape[0]

        l_sum = np.sum(image[int(3 * image_height / 4):, :int(image_width / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - self.window_width / 2
        r_sum = np.sum(image[int(3 * image_height / 4):, int(image_width / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - self.window_width / 2 + int(image_width / 2)

        window_centroids.append((l_center, r_center))

        for level in range(1, (int)(image_height / self.window_height)):
            image_layer = np.sum(
                image[int(image_height - (level + 1) * self.window_height): int(image_height - level * self.window_height), :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)
            offset = self.window_width / 2
            l_min_index = int(max(l_center + offset - self.margin, 0))
            l_max_index = int(min(l_center + offset + self.margin, image_width))
            l_center = np.argmax(conv_signal[l_min_index: l_max_index]) + l_min_index - offset

            r_min_index = int(max(r_center + offset - self.margin, 0))
            r_max_index = int(min(r_center + offset + self.margin, image_width))
            r_center = np.argmax(conv_signal[r_min_index: r_max_index]) + r_min_index - offset

            window_centroids.append((l_center, r_center))

        self.recent_centroids.append(window_centroids)
        #self.recent_centroids = self.recent_centroids[-self.smooth_factor:]
        return np.average(self.recent_centroids[-self.smooth_factor:], axis=0)