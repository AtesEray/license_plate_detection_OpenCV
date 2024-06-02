import cv2
import numpy as np
import imutils
import easyocr
import os

# Constants
IMG_PATH = "plaka_okuma/data/licence_plate.jpg"
LOW_THRESHOLD = 110
HIGH_THRESHOLD = 250
MIN_AREA = 15
MAX_AREA = 600
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
SHARPENING_WEIGHT = 1.5
BLURRING_WEIGHT = -0.5

# Helper functions
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (720, 720))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray, 11, 17, 17)
    return img, img_gray, img_blur

def find_contours(canny_img):
    contours = cv2.findContours(canny_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return sorted(contours, key=cv2.contourArea, reverse=True)[:10]

def find_screen_contour(contours):
    for cnt in contours:
        epsilon = 0.018 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            return approx
    return None

def create_mask_from_contour(img_shape, contour):
    mask = np.zeros(img_shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    return mask

def crop_to_contour(img_gray, mask):
    x, y = np.where(mask == 255)
    if not x.size or not y.size:
        print("No white pixels found in the mask.")
        return None
    topx, topy = np.min(x), np.min(y)
    bottomx, bottomy = np.max(x), np.max(y)
    return img_gray[topx:bottomx + 1, topy:bottomy + 1]

def apply_clahe(cropped_img):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(cropped_img)

def sharpen_image(contrasted_img, median_filtered_img):
    return cv2.addWeighted(contrasted_img, SHARPENING_WEIGHT, median_filtered_img, BLURRING_WEIGHT, 0)

def blob_coloring(binary_img):
    num_labels, labels_im = cv2.connectedComponents(binary_img)
    label_hue = np.uint8(179 * labels_im / np.max(labels_im))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    return labeled_img, num_labels, labels_im

def separate_text(binary_img, num_labels, labels_im):
    output = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        mask = (labels_im == i).astype("uint8") * 255
        area = cv2.countNonZero(mask)
        if MIN_AREA < area < MAX_AREA:
            output = cv2.bitwise_or(output, mask)
    return output

def read_number(roi_img):
    reader = easyocr.Reader(['en'])
    return reader.readtext(roi_img)

def show_images(*args):
    titles = ["Cropped", "HighContrast", "MedianFiltered", "Sharped", "BlobColoring", "BinaryRoi", "SeparatedBlob", "CannyEdge"]
    for img, title in zip(args, titles):
        cv2.imshow(title, img)

def write_images(output_path, file_name, *args):
    titles = ["Cropped", "HighContrast", "MedianFiltered", "Sharped", "CannyEdge", "Original", "Gray", "BlobColoring", "BinaryRoi", "SeparatedBlob"]
    for img, title in zip(args, titles):
        cv2.imwrite(f"{output_path}/{file_name}_{title}X.png", img)

def main():
    base_name = os.path.basename(IMG_PATH)
    file_name, _ = os.path.splitext(base_name)
    output_path = "plaka_okuma\deneme"

    img, img_gray, img_blur = load_and_preprocess_image(IMG_PATH)
    canny = cv2.Canny(img_blur, LOW_THRESHOLD, HIGH_THRESHOLD)
    contours = find_contours(canny)
    screen_contour = find_screen_contour(contours)

    if screen_contour is None:
        print("Error: No screen contour detected.")
        return

    mask = create_mask_from_contour(img_gray.shape, screen_contour)
    cropped = crop_to_contour(img_gray, mask)

    if cropped is None:
        print("Error: No cropped area detected.")
        return

    contrasted_roi = apply_clahe(cropped)
    median_filtered_roi = cv2.medianBlur(contrasted_roi, 3)
    sharp_roi = sharpen_image(contrasted_roi, median_filtered_roi)

    _, binary_roi = cv2.threshold(sharp_roi, 60, 255, cv2.THRESH_BINARY_INV)
    blobed_img, num_labels, labels_im = blob_coloring(binary_roi)
    separated_roi = separate_text(binary_roi, num_labels, labels_im)
    ocr_results = read_number(separated_roi)

    show_images(cropped, contrasted_roi, median_filtered_roi, sharp_roi, blobed_img, binary_roi, separated_roi, canny)
    write_images(output_path, file_name, cropped, contrasted_roi, median_filtered_roi, sharp_roi, canny, img, img_gray, blobed_img, binary_roi, separated_roi)

    for bbox, text, prob in ocr_results:
        if float(prob) >= 0.4:
            print(f'Text: {text}, Probability: {prob}')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()