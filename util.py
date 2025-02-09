import pickle

from skimage.transform import resize
import numpy as np
import cv2


EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("model.p", "rb"))


def empty_or_not(spot_bgr):

    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


# def get_parking_spots_bboxes(connected_components):
#     (totalLabels, label_ids, values, centroid) = connected_components

#     slots = []
#     coef = 1
#     for i in range(1, totalLabels):

#         # Now extract the coordinate points
#         x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
#         y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
#         w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
#         h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

#         slots.append([x1, y1, w, h])

#     return slots
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

     # Define entry and exit points
    entry1 = (95,58)
    entry2 = (1464, 64)
    exit_point = (109, 1074)

    slots = []
    coef = 1
    for i in range(1, totalLabels):
        # Extract the bounding box coordinates
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        
        # Calculate the center of the bounding box
        center_x = x1 + w // 2
        center_y = y1 + h // 2

        # Calculate distances to entry and exit points
        d1 = np.sqrt((center_x - entry1[0])**2 + (center_y - entry1[1])**2)
        d2 = np.sqrt((center_x - entry2[0])**2 + (center_y - entry2[1])**2)
        d3 = np.sqrt((center_x - exit_point[0])**2 + (center_y - exit_point[1])**2)

        # Append bounding box and distances
        slots.append([x1, y1, w, h, d1, d2, d3])


    return slots

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# def calc_diff(prev_im, curr_im, stride=2):
#     # Sample pixels using strides
#     sampled_curr = curr_im[::stride, ::stride]
#     sampled_prev = prev_im[::stride, ::stride]

#     # Compute the absolute mean difference of sampled pixels
#     diff = np.abs(sampled_prev - sampled_curr)
#     return np.mean(diff)
