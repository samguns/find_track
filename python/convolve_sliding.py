import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

warped = mpimg.imread("warped_example.jpg")
window_width = 50
window_height = 80
margin = 100


def window_mask(width, height, img_ref, center, level):
    image_width = img_ref.shape[1]
    image_height = img_ref.shape[0]
    output = np.zeros_like(img_ref)
    output[int(image_height - (level+1)*height) : int(image_height - level*height),
           max(0, int(center - width/2)) : min(int(center + width/2), image_width)] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = []
    window = np.ones(window_width)
    image_width = image.shape[1]
    image_height = image.shape[0]

    l_sum = np.sum(image[int(3*image_height/4):, :int(image_width/2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2
    r_sum = np.sum(image[int(3*image_height/4):, int(image_width/2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 + int(image_width/2)

    window_centroids.append((l_center, r_center))

    for level in range(1, (int)(image_height / window_height)):
        image_layer = np.sum(image[int(image_height - (level+1) * window_height): int(image_height - level * window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        offset = window_width / 2
        l_min_index = int(max(l_center - offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image_width))
        l_center = np.argmax(conv_signal[l_min_index : l_max_index]) + l_min_index - offset

        r_min_index = int(max(r_center - offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image_width))
        r_center = np.argmax(conv_signal[r_min_index : r_max_index]) + r_min_index - offset

        window_centroids.append((l_center, r_center))

    return window_centroids


window_centroids = find_window_centroids(warped, window_width, window_height, margin)

if len(window_centroids) > 0:
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    for level in range(0, len(window_centroids)):
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    template = np.array(r_points + l_points, np.uint8)
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    warpage = np.dstack((warped, warped, warped)) * 255
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

else:
    output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

leftx = []
rightx = []
for level in range(0, len(window_centroids)):
    leftx.append(window_centroids[level][0])
    rightx.append(window_centroids[level][1])

warped_height = warped.shape[0]
yvals = range(0, warped_height)
res_yvals = np.arange(warped_height-(window_height/2), 0, -window_height)
left_fit = np.polyfit(res_yvals, leftx, 2)

left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
left_fitx = np.array(left_fitx, np.int32)

plt.imshow(output)
plt.show()