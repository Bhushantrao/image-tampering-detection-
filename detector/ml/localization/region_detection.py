import cv2
import numpy as np

def localize_region(original_path,
                     heatmap_path,
                     output_path):

    image = cv2.imread(
        original_path
    )

    heatmap = cv2.imread(
        heatmap_path,
        0
    )

    # Threshold suspicious area
    _, threshold = cv2.threshold(
        heatmap,
        150,
        255,
        cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Ignore very small regions
        if area > 100:

            x,y,w,h = cv2.boundingRect(
                contour
            )

            cv2.rectangle(
                image,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )

    cv2.imwrite(
        output_path,
        image
    )

    return output_path