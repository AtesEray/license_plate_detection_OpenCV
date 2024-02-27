import cv2
import numpy as np
from PIL import Image
import pytesseract
import imutils

path= "plaka_okuma\data\WhatsApp Image 2024-02-27 at 14.10.30.jpeg"

img = cv2.imread(path)

img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
img_blur = cv2.bilateralFilter(img_gray,5,250,250)
img_gaussian = blur = cv2.GaussianBlur(img_gray, (7, 5), 0)
canny= cv2.Canny(img_blur, 100,255)

contours= cv2.findContours(canny, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(contours)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screen = None

for cnt in cnts:

    #arclength yay uzunlugu demek
    epsilon = 0.018 * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt , epsilon, True)

    

    #KODU KAMERANIN KONUMUNA GÖRE ÖZELLİŞTİRMEK BAŞARIYI ÇOK DAHA ARTTIRIR KAMERA KONUMUNA GÖRE BELLİ BİR ROİ İÇERİSİNDEN TEPSİT YAPTIRMAYI DÜŞÜNÜYORUM.
    area = cv2.contourArea(approx)
    if len(approx) == 4 and area > 1000 and area < 5000 :
        screen = approx
        print(area)
        break
if  screen is not None:
    mask = np.zeros(img_gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screen], 0, (255,255,255), -1)
    new_image = cv2.bitwise_and(img,img, mask = mask)

    (x,y) = np.where(mask == 255)
    (topx , topy) = (np.min(x), np.min(y))
    (bottomx , bottomy) = (np.max(x), np.max(y))

    cropped = img_gray[topx: bottomx  +1, topy: bottomy+1]

    txt = pytesseract.image_to_string(cropped, lang = "eng")
    print(txt)
else:
    print("hata")
    
    


    


cv2.imshow("original",img)
cv2.imshow("blur",img_blur)

cv2.imshow("canny",canny)
if  screen is not None:
    cv2.imshow("new",new_image)
    cv2.imshow("mask",cropped)


cv2.waitKey(0)
cv2.destroyAllWindows()
