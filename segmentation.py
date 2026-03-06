# segmentation.py

import cv2
import numpy as np
import os


def segment_characters(processed_image_path, output_folder):

    image = cv2.imread(processed_image_path)
    cv2.imwrite("debug_input.jpg", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OTSU threshold
    _, thresh1 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove tiny white noise dots
    kernel_small = np.ones((3,3), np.uint8)
    clean = cv2.medianBlur(thresh1, 5)

    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_small)

    # reconnect small character gaps
    kernel_close = np.ones((4,4), np.uint8)
    dilate = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close)

    cv2.imwrite("debug_thresh.jpg", dilate)

    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dilate, connectivity=8
    )

    valid_components = []

    for i in range(1, num_labels):  # skip background

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Filter components by size
        if area > 80 and area < 5000:
            valid_components.append((x, y, w, h))

    print("Valid character components:", len(valid_components))

    # sort characters left-to-right, top-to-bottom
    sorted_ctrs = sorted(
        valid_components,
        key=lambda b: b[0] + b[1] * image.shape[1]
    )

    orig = image.copy()

    os.makedirs(output_folder, exist_ok=True)

    roi_paths = []
    i = 0

    for (x, y, w, h) in sorted_ctrs:

        roi = image[y:y+h, x:x+w]

        cv2.rectangle(orig, (x, y), (x+w, y+h), (0,255,0), 2)

        roi_path = os.path.join(output_folder, "roi" + str(i) + ".png")

        cv2.imwrite(roi_path, roi)

        roi_paths.append(roi_path)

        i += 1

    box_path = os.path.join(output_folder, "box.jpg")
    cv2.imwrite(box_path, orig)

    return roi_paths
