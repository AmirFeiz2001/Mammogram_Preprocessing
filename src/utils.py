import cv2
import numpy as np

def paint_abnormality(image, chain_code, color_value=255, width=5):

    image_copy = image.copy()
    column, row = int(chain_code[0]), int(chain_code[1])
    image_copy[row, column] = color_value

    for code in chain_code[2:]:
        if code == "0": row -= 1
        elif code == "1": row -= 1; column += 1
        elif code == "2": column += 1
        elif code == "3": row += 1; column += 1
        elif code == "4": row += 1
        elif code == "5": row += 1; column -= 1
        elif code == "6": column -= 1
        elif code == "7": row -= 1; column -= 1
        for w in range(width):
            if 0 <= row + w < image_copy.shape[0] and 0 <= column < image_copy.shape[1]:
                image_copy[row + w, column] = color_value
    return image_copy

def load_chaincode(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i >= 8:  # Chaincode starts at line 9
                return line.strip().split()
    raise ValueError("Chaincode not found in file")
