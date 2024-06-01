import cv2
import numpy as np
from PIL import Image
import imutils
import easyocr
import os

img_path =  "plaka_okuma\data\licence_plate.jpg"

base_name = os.path.basename(img_path)
file_name, file_ext = os.path.splitext(base_name)

def blob_coloring(sharp_roi) :
    
    # Eşikleme (Thresholding) ile ikili (binary) görüntü elde edin
    _, binary_roi = cv2.threshold(sharp_roi, 60, 255, cv2.THRESH_BINARY_INV)


    # Bağlantı bileşeni analizi (Blob coloring)
    num_labels, labels_im = cv2.connectedComponents(binary_roi)

    # Her bir bileşeni farklı bir renkle işaretleyin
    label_hue = np.uint8(179 * labels_im / np.max(labels_im))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    return labeled_img, binary_roi , num_labels , labels_im

def seperateTxt (binary_roi , num_labels , labels_im):
    # Boş bir görüntü oluşturur (aynı boyutta, siyah)
    output = np.zeros_like(binary_roi)
    
    min_area = 15  # Bu değeri manuel testler sonucu optimize ettik
    max_area = 600  # Bu değeri manuel testler sonucu optimize ettik
    for i in range(1, num_labels):  # 0 etiketi arka plan olduğu için 1'den başlıyoruz
        mask = (labels_im == i).astype("uint8") * 255
        area = cv2.countNonZero(mask)
        if min_area < area < max_area:
            output = cv2.bitwise_or(output, mask)
    return output

def readingNumber (seperated_roi ):
    # OCR ile karakterleri okuyun
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
        print("Deneme")
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_Cropped.png", cropped)
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_HighContrast.png", contrasted_roi)
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_median_filtered.png", median_filtered_roi)
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_Sharped.png", sharp_roi)
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_Canny_Edge.png", canny)
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_Original.png", img)
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_Gray.png", img_gray)
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_BlobeColoring.png", blobedImg)
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_BinaryRoi.png", binary_roi)
        cv2.imwrite(f"plaka_okuma\Results/{file_name}_SeperatedBlobed.png", seperatedRoi)

def main():

    img = cv2.imread(img_path)
    img = cv2.resize(img, (720,720))
    # Griye çevirme işlemi yapılıyor
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img_blur = cv2.bilateralFilter(img_gray,11,17,17)

    #Kenarları bulmak için canny uygulanıyor
    low_threshold = 110
    high_threshold = 250
    canny= cv2.Canny(img_blur, low_threshold,high_threshold)


    # cv2.findContours fonksiyonu, 'canny' görüntüsündeki kenarları tespit eder ve bu kenarları kontur olarak döner.
    # cv2.CHAIN_APPROX_SIMPLE parametresi, kontur noktalarının sayısını azaltarak yalnızca gerekli noktaları tutar.
    # Sonuç olarak, 'contours' değişkeni konturların bir listesini ve hiyerarşik ilişkilerini içerir.
    contours= cv2.findContours(canny.copy(), cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

    # imutils.grab_contours fonksiyonu, contours listesindeki konturları alır.
    # Bu konturları cv2.contourArea fonksiyonunu kullanarak alanlarına göre azalan sırada sıralar ve ilk 10 tanesini seçer.
    # Sonuç olarak, en büyük 10 konturu 'cnts' listesinde saklar.
    cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    screen = None
    cropped = None
    for cnt in cnts:

        # epsilon, konturun çevresinin %1.8'i olarak hesaplanır. Bu, konturun yaklaştırılması (approximation) için hassasiyet değeridir.
        epsilon = 0.018 * cv2.arcLength(cnt,True)
        # cv2.approxPolyDP fonksiyonu, epsilon hassasiyetine göre konturu daha az noktayla temsil eder.
        approx = cv2.approxPolyDP(cnt ,10, True)

        if len(approx) == 4 :
            screen = approx
            break

    if  screen is not None:
        mask = np.zeros(img_gray.shape, np.uint8)
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
        print("hata")

        
    # Kontrastı artırmak için CLAHE kullandik
    if cropped is  None:
        print("Tespit basarisiz")
        return 0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted_roi = clahe.apply(cropped)

    # contrasted_roi uzerine medianBlur uygulanarak gurultu azaltilir. 
    median_filtered_roi = cv2.medianBlur(contrasted_roi, 3)

    #Goruntuye daha da cok keskinlik kazadirildi
    sharp_roi = cv2.addWeighted(contrasted_roi,1.5 , median_filtered_roi, -0.5,0)


    if  screen is not None:


        #Blob colorin fonksiyonu uygulaniyor
        blobedImg , binary_roi , num_labels , labels_im = blob_coloring(sharp_roi=sharp_roi)


        #Blob coloring sonrasi yazilari ayirmak isin seperateTxt fonksiyonu aktive edildi
        seperatedRoi = seperateTxt(binary_roi= binary_roi , num_labels= num_labels , labels_im= labels_im)

        #Goruntunun son halinde yazi tespiti gerceklesiyor
        result = readingNumber(seperated_roi=seperatedRoi)

        showImage(cropped=cropped , contrasted_roi=contrasted_roi, median_filtered_roi=median_filtered_roi,sharp_roi=sharp_roi, blobedImg=blobedImg, binary_roi=binary_roi,seperatedRoi=seperatedRoi, canny=canny)
        writeImage(cropped=cropped , contrasted_roi=contrasted_roi, median_filtered_roi=median_filtered_roi,sharp_roi=sharp_roi, blobedImg=blobedImg, binary_roi=binary_roi,seperatedRoi=seperatedRoi, img=img, img_gray=img_gray, canny=canny)
        

        # OCR sonuçlarını yazar

        for (bbox, text, prob) in result:
            if float(prob) >= 0.4:
                print(f'Text: {text}, Probability: {prob}')

        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()