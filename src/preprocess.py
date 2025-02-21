import cv2
import numpy as np

class MammogramPreprocess:
    
    def __init__(self, image, chaincode, flag, width, height):
        self._image = image
        self._org_image = image.copy()
        self._chaincode = chaincode
        self._flag = flag
        self._width = width
        self._height = height

    def _crop_image(self):
        #Crop black borders and non-breast regions from the image.
        image = self._image
        columns = int(image.shape[0] * 0.09)  # 9% from black side
        columns_breast_side = int(image.shape[1] * 0.025)  # 2.5% from breast side
        image = np.transpose(image)
        unique, counts = np.unique(image[:columns], return_counts=True)
        freq = dict(zip(unique, counts))

        # Determine breast orientation
        if freq[min(freq)] > freq[max(freq)] + 150000:  # Breast to the left
            image = image[columns:-columns_breast_side]
        else:  # Breast to the right
            image = image[columns_breast_side:-columns]

        image = np.transpose(image)
        rows = int(image.shape[0] * 0.03)
        image = image[rows:-rows]
        rows = int(image.shape[0] * 0.015)
        return image[rows:-rows]

    def _remove_text(self):
        #Remove text artifacts from the image
        image = self._image
        ret, thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
        kernel = np.ones((30, 30), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((550, 550), np.uint8)
        erosion = cv2.erode(opening, kernel, iterations=1)
        kernel = np.ones((800, 800), np.uint8)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        kernel = np.ones((220, 220), np.uint8)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        kernel = np.ones((600, 600), np.uint8)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        return cv2.bitwise_and(image, image, mask=dilation)

    def _remove_side_effect1(self):
        #Remove side effects after single CLAHE.
        image = self._image
        ret, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((30, 30), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((50, 50), np.uint8)
        erosion = cv2.erode(opening, kernel, iterations=1)
        kernel = np.ones((120, 120), np.uint8)
        dilation = cv2.dilate(erosion, kernel, iterations=2)
        return cv2.bitwise_and(image, thresh, mask=dilation)

    def _remove_side_effect2(self):
        image = self._image
        ret, thresh = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
        kernel = np.ones((30, 30), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((50, 50), np.uint8)
        erosion = cv2.erode(opening, kernel, iterations=1)
        kernel = np.ones((120, 120), np.uint8)
        dilation = cv2.dilate(erosion, kernel, iterations=3)
        return cv2.bitwise_and(image, thresh, mask=dilation)

    def _enhance_image(self):
        image = self._image
        if self._flag == 1:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
            self._image = clahe.apply(image)
            return self._remove_side_effect1()
        elif self._flag == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
            clahe_image = clahe.apply(image)
            clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
            self._image = clahe2.apply(clahe_image)
            return self._remove_side_effect2()
        else:
            raise ValueError("Invalid flag: Use 1 for single CLAHE, 2 for double CLAHE")

    def _change_image_direction_to_right(self):
        image = self._image
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        thresh = np.transpose(thresh)
        unique, counts = np.unique(thresh[:int(thresh.shape[0] * 0.1)], return_counts=True)
        freq = dict(zip(unique, counts))
        return 1 if freq[min(freq)] >= freq[max(freq)] else -1

    def _adjust_chaincode(self, change_or_not):
        org_chaincode = self._chaincode
        column, row = self._org_image.shape[0], self._org_image.shape[1]
        new_chaincode = list(org_chaincode)

        cropped_column = int(column * 0.09) if change_or_not == 1 else int(column * 0.025)
        cropped_row = int(row * 0.06) + int(row * 0.015)

        new_chaincode[0] = str(int(org_chaincode[0]) - cropped_column)
        new_chaincode[1] = str(int(org_chaincode[1]) - cropped_row)
        return new_chaincode

    def _mirror_chaincode(self, change_or_not):
        if change_or_not == 1:
            new_chaincode = ["res", "res", ""]
            for code in self._chaincode[2:]:
                mapping = {"1": "7", "7": "1", "2": "6", "6": "2", "3": "5", "5": "3", "0": "0", "4": "4"}
                new_chaincode.append(mapping.get(code, code))
            new_chaincode[0] = str(self._image.shape[0] - int(self._chaincode[0]))
            new_chaincode[1] = self._chaincode[1]
            return new_chaincode
        return self._chaincode

    def _flip_if_needed(self, change_or_not):
        return cv2.flip(self._image, 1) if change_or_not == 1 else self._image

    def _get_4_corners_of_chaincode(self):
        chaincode = self._chaincode
        column, row = int(chaincode[0]), int(chaincode[1])
        min_col, min_row, max_col, max_row = float('inf'), float('inf'), 0, 0

        for code in chaincode[2:]:
            if code == "0": row -= 1
            elif code == "1": row -= 1; column += 1
            elif code == "2": column += 1
            elif code == "3": row += 1; column += 1
            elif code == "4": row += 1
            elif code == "5": row += 1; column -= 1
            elif code == "6": column -= 1
            elif code == "7": row -= 1; column -= 1
            min_col, min_row = min(min_col, column), min(min_row, row)
            max_col, max_row = max(max_col, column), max(max_row, row)

        return min_col, min_row, max_col, max_row

    def _scale_bounding_box(self, minx, miny, maxx, maxy):
        x_scale, y_scale = self._height / self._image.shape[0], self._width / self._image.shape[1]
        return [
            int(np.round(minx * y_scale)),
            int(np.round(miny * x_scale)),
            int(np.round(maxx * y_scale)),
            int(np.round(maxy * x_scale))
        ]

    def plot_boundingbox(self, scaled_boundingbox):
        start_point = (scaled_boundingbox[0], scaled_boundingbox[1])
        end_point = (scaled_boundingbox[2], scaled_boundingbox[3])
        return cv2.rectangle(self._image, start_point, end_point, (255, 0, 0), 3)

    def preprocess(self):
        self._image = self._crop_image()
        self._image = self._remove_text()
        self._image = self._enhance_image()
        change_or_not = self._change_image_direction_to_right()
        self._chaincode = self._adjust_chaincode(change_or_not)
        self._chaincode = self._mirror_chaincode(change_or_not)
        self._image = self._flip_if_needed(change_or_not)
        miny, minx, maxy, maxx = self._get_4_corners_of_chaincode()
        self._image = cv2.resize(self._image, (self._width, self._height), interpolation=cv2.INTER_AREA)
        scaled_bounding_box = self._scale_bounding_box(minx, miny, maxx, maxy)
        return self._image, scaled_bounding_box
