import cv2
import numpy as np

def paint_abnormality(image, chain_code, color_value=255, width=5):
    """
    Paint abnormality contour on the image using chaincode.

    Args:
        image (np.ndarray): Input image.
        chain_code (list): Chaincode for abnormality.
        color_value (int): Color intensity for painting.
        width (int): Width of the painted line.

    Returns:
        np.ndarray: Image with painted abnormality.
    """
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
    """
    Load chaincode from an overlay file.

    Args:
        file_path (str): Path to the overlay file.

    Returns:
        list: Chaincode as a list of strings.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i >= 8:  # Chaincode starts at line 9
                return line.strip().split()
    raise ValueError("Chaincode not found in file")
