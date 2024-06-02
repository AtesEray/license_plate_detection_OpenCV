import cv2
import numpy as np
import imutils
import easyocr
import os

img_path =  "plaka_okuma/data/licence_plate.jpg"

base_name = os.path.basename(img_path)
file_name, file_ext = os.path.splitext(base_name)

def blob_coloring(sharp_roi) :
    
    # Obtain binary images with Thresholding
    _, binary_roi = cv2.threshold(sharp_roi, 60, 255, cv2.THRESH_BINARY_INV)


    # Connection component analysis (Blob coloring)
    num_labels, labels_im = cv2.connectedComponents(binary_roi)

    # Mark each component with a different color
    label_hue = np.uint8(179 * labels_im / np.max(labels_im))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    return labeled_img, binary_roi , num_labels , labels_im

def seperateTxt (binary_roi , num_labels , labels_im):
    # Creates a blank image (same size, black)
    output = np.zeros_like(binary_roi)
    
    min_area = 15  # We optimized this value as a result of manual tests.
    max_area = 600  # We optimized this value as a result of manual tests.
    for i in range(1, num_labels):  # Since label 0 is the background, we start from 1
        mask = (labels_im == i).astype("uint8") * 255
        area = cv2.countNonZero(mask)
        if min_area < area < max_area:
            output = cv2.bitwise_or(output, mask)
    return output

def readingNumber (seperated_roi ):
    # Read characters with OCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(seperated_roi)
    return result

def showImage(cropped, contrasted_roi,median_filtered_roi,sharp_roi,blobedImg,binary_roi,seperatedRoi, canny):
        cv2.imshow("Cropped",cropped)
        cv2.imshow("HighContrast", contrasted_roi)
        cv2.imshow("median_filtered", median_filtered_roi)
        cv2.imshow("Sharped", sharp_roi)
        cv2.imshow("BlobeColoring", blobedImg)
        cv2.imshow("BinaryRoi", binary_roi)
        cv2.imshow("SeperatedBlobed" , seperatedRoi)
        cv2.imshow("CannyEdge", canny)

def writeImage(cropped, contrasted_roi,median_filtered_roi,canny,blobedImg,binary_roi,seperatedRoi,img,img_gray , sharp_roi):
        cv2.imwrite(f"plaka_okum/Rapor_Tespit/{file_name}_CroppedX.png", cropped)
        cv2.imwrite(f"plaka_okuma/Rapor_Tespit/{file_name}_HighContrastX.png", contrasted_roi)
        cv2.imwrite(f"plaka_okuma/Rapor_Tespit/{file_name}_median_filteredX.png", median_filtered_roi)
        cv2.imwrite(f"plaka_okuma/Rapor_Tespit/{file_name}_SharpedX.png", sharp_roi)
        cv2.imwrite(f"plaka_okuma/Rapor_Tespit/{file_name}_Canny_EdgeX.png", canny)
        cv2.imwrite(f"plaka_okuma/Rapor_Tespit/{file_name}_OriginalX.png", img)
        cv2.imwrite(f"plaka_okuma/Rapor_Tespit/{file_name}_GrayX.png", img_gray)
        cv2.imwrite(f"plaka_okuma/Rapor_Tespit/{file_name}_BlobeColoringX.png", blobedImg)
        cv2.imwrite(f"plaka_okuma/Rapor_Tespit/{file_name}_BinaryRoiX.png", binary_roi)
        cv2.imwrite(f"plaka_okuma/Rapor_Tespit/{file_name}_SeperatedBlobedX.png", seperatedRoi)


def main():

    img = cv2.imread(img_path)
    img = cv2.resize(img, (720,720))
    # Conversion to gray is in progress
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray,11,17,17)

    #Applying canny to find edges
    low_threshold = 110
    high_threshold = 250
    canny= cv2.Canny(img_blur, low_threshold,high_threshold)


    # The cv2.findContours function detects the edges in the 'canny' image and returns these edges as contours.
    # The cv2.CHAIN_APPROX_SIMPLE parameter reduces the number of contour points, keeping only the necessary points.
    # As a result, the 'contours' variable contains a list of contours and their hierarchical relationships.
    contours= cv2.findContours(canny.copy(), cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

    # The imutils.grab_contours function grabs the contours from the contours list.
    # It sorts these contours in descending order according to their area using the cv2.contourArea function and selects the first 10 of them.
    # As a result, it stores the 10 largest contours in the 'cnts' list.
    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    screen = None
    cropped = None
    for cnt in cnts:

        # epsilon is calculated as 1.8% of the perimeter of the contour. This is the precision value for approximation of the contour.
        epsilon = 0.018 * cv2.arcLength(cnt,True)
        # The cv2.approxPolyDP function represents the contour with fewer points according to epsilon precision.
        approx = cv2.approxPolyDP(cnt ,10, True)

        if len(approx) == 4 :
            screen = approx
            break

    if  screen is not None:

        #We crated mask which shape in img_gray
        mask = np.zeros(img_gray.shape, np.uint8)

        #Drawing contour to the mask from screen values
        cv2.drawContours(mask, [screen], -1, (255), thickness=cv2.FILLED)

        (x,y) = np.where(mask == 255)

        if x.size == 0 or y.size == 0:
            print("No white pixels found in the mask.")
            # Handle the case when no white pixels are found
            # For example, you can skip the cropping step or set a default cropped region
            cropped = img_gray  # or set a default region
        else: 
            (topx , topy) = (np.min(x), np.min(y))
            (bottomx , bottomy) = (np.max(x), np.max(y))
            cropped = img_gray[topx: bottomx  +1, topy: bottomy+1]
    else:
        print("Error")

        
    if cropped is  None:
        print("No cropped detected")
        return 0
    
    # Used CLAHE to increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted_roi = clahe.apply(cropped)

    # Noise is reduced by applying medianBlur over the contrast-enhanced roi. 
    median_filtered_roi = cv2.medianBlur(contrasted_roi, 3)

    #Added even more sharpness to the image
    sharp_roi = cv2.addWeighted(contrasted_roi,1.5 , median_filtered_roi, -0.5,0)


    if  screen is not None:

        #Blob colorin function is applied
        blobedImg , binary_roi , num_labels , labels_im = blob_coloring(sharp_roi=sharp_roi)

        #The separateTxt function was activated to separate the texts after blob coloring.
        seperatedRoi = seperateTxt(binary_roi= binary_roi , num_labels= num_labels , labels_im= labels_im)

        #Text detection is performed in the final version of the image
        result = readingNumber(seperated_roi=seperatedRoi)

        showImage(cropped=cropped , contrasted_roi=contrasted_roi, median_filtered_roi=median_filtered_roi,sharp_roi=sharp_roi, blobedImg=blobedImg, binary_roi=binary_roi,seperatedRoi=seperatedRoi, canny=canny)
        writeImage(cropped=cropped , contrasted_roi=contrasted_roi, median_filtered_roi=median_filtered_roi,sharp_roi=sharp_roi, blobedImg=blobedImg, binary_roi=binary_roi,seperatedRoi=seperatedRoi, img=img, img_gray=img_gray, canny=canny)
        
        # Writes OCR results
        for (bbox, text, prob) in result:
            if float(prob) >= 0.4:
                print(f'Text: {text}, Probability: {prob}')

        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()