import numpy as np


class LaneIdentifier:
    def __init__(self, smooth_factor):
        self.margin = 100
        self.smooth_factor = smooth_factor
        self.window_width = 50
        self.window_height = 80
        self.recent_centroids = []
        self.left_fit = None
        self.right_fit = None
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
            self.left_fit = np.polyfit(res_yvals, leftx, 2)
            self.right_fit = np.polyfit(res_yvals, rightx, 2)

            return True, self.left_fit, self.right_fit

        return False, None, None

    def reject_anomaly(self, y, l_center, r_center):
        if self.left_fit is not None:
            last_l_center = self.left_fit[0]*y*y + self.left_fit[1]*y + self.left_fit[2]
            if l_center < (last_l_center - self.margin) or \
                l_center > (last_l_center + self.margin):
                l_center = last_l_center

        if self.right_fit is not None:
            last_r_center = self.right_fit[0]*y*y + self.right_fit[1]*y + self.right_fit[2]
            if r_center < (last_r_center - self.margin) or \
                r_center > (last_r_center + self.margin):
                r_center = last_r_center

        return l_center, r_center

    def find_window_centroids(self, image):
        window_centroids = []
        window = np.ones(self.window_width)
        image_width = image.shape[1]
        image_height = image.shape[0]
        offset = self.window_width / 2

        l_sum = np.sum(image[int(3 * image_height / 4):, :int(image_width / 2)], axis=0)
        l_center = np.argmax(np.convolve(l_sum, window)) - self.window_width / 2
        r_sum = np.sum(image[int(3 * image_height / 4):, int(image_width / 2):], axis=0)
        r_center = np.argmax(np.convolve(r_sum, window)) - self.window_width / 2 + int(image_width / 2)

        y = image_height - (self.window_height / 2)
        l_center, r_center = self.reject_anomaly(y, l_center, r_center)

        window_centroids.append((l_center, r_center))

        for level in range(1, (int)(image_height / self.window_height)):
            image_layer = np.sum(
                image[int(image_height - (level + 1) * self.window_height): int(image_height - level * self.window_height), :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)

            l_min_index = int(max(l_center - offset - self.margin, 0))
            l_max_index = int(min(l_center + offset + self.margin, image_width))
            det_l_center = np.argmax(conv_signal[l_min_index: l_max_index]) + l_min_index - offset
            if abs(det_l_center - l_center) < offset:
                l_center = det_l_center

            r_min_index = int(max(r_center - offset - self.margin, 0))
            r_max_index = int(min(r_center + offset + self.margin, image_width))
            det_r_center = np.argmax(conv_signal[r_min_index: r_max_index]) + r_min_index - offset
            if abs(det_r_center - r_center) < offset:
                r_center = det_r_center

            y -= self.window_height
            l_center, r_center = self.reject_anomaly(y, l_center, r_center)

            window_centroids.append((l_center, r_center))

        self.recent_centroids.append(window_centroids)
        return np.average(self.recent_centroids[-self.smooth_factor:], axis=0)