import cv2
import numpy as np
import copy

def encode(img: np.ndarray, qr: np.ndarray, opacity: float) -> np.ndarray:
    encoded = copy.deepcopy(img)
    qr = cv2.resize(qr, (encoded.shape[1], encoded.shape[0]), interpolation = cv2.INTER_AREA)
    encoded = cv2.addWeighted(encoded, 1 - opacity, qr, opacity, 0)
    return encoded

def decode(img: np.ndarray, encoded: np.ndarray, opacity: float) -> np.ndarray:
    
    return (cv2.addWeighted(encoded, 1, img, -(1 - opacity), 0)) * 255

def denoise(img: np.ndarray) -> np.ndarray:
    img = cv2.medianBlur(img, 35)
    (b, g, r) = cv2.split(img)
    _, thresh1 = cv2.threshold(b, 1, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(g, 1, 255, cv2.THRESH_BINARY)
    _, thresh3 = cv2.threshold(r, 1, 255, cv2.THRESH_BINARY)
    img = thresh1 & thresh2 & thresh3

    erosion_kernel = np.ones((25, 25), np.uint8)
    dilation_kernel = np.ones((18, 18), np.uint8)

    img = cv2.erode(img, erosion_kernel, 1)
    img = cv2.dilate(img, dilation_kernel, 1)

    img = cv2.resize(img, (max(img.shape[1], img.shape[0]),) * 2, interpolation = cv2.INTER_AREA)
    
    return img