import random

import numpy as np
import cv2 as cv


def find_matches(images, pattern):
    result = []

    for image in images:
        score = random.random()
        is_match = score >= 0.5
        image_result = dict(value=score, is_match=is_match)
        result.append(image_result)

    return result
