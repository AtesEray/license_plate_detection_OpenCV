# license_plate_detection_OpenCV
license plate detection 

# Version 1

Currently, the project is in its initial version. However, the license plate detection feature is not yet fully functional. In upcoming versions, I am actively working on improving the license plate detection algorithm to adapt to various scenarios, thereby increasing the success rate

#What can you do with version 1 ?

*You can easily identify the license plates of vehicles taken from a proper and close angle.
*It currently does not work effectively for photos taken from a distance and at wrong angles.


# Version 2

The 2nd version of the License Plate Reading system has been released.

Many additional functions have been added to the previous version. Many of these were used to separate the text in the detected license plate area.

#What is new in Version 2?

-By using blurring and sharpening processes more effectively, the detection success of the license plate area has been increased.
-By applying "blob coloring", the letters on the plate can now be successfully separated from the entire picture. Our text detection operations are carried out much more accurately than in the previous version.
-"EasyOCR" was started to be used instead of "Pytesseract" library and much more efficient results were obtained.

#What is next?

My next goal is to have it check whether the detected license plates are registered in the system. I plan to develop a system with a simple Python interface that will check the registration of the incoming vehicle into the system and check whether it will be allowed to pass.
