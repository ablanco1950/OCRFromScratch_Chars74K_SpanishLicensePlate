# OCRFromScratch_Chars74K_SpanishLicensePlate
OCR from scratch using Chars74 Dataset: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ applied to the case of Spanish car license plates   or any other with format NNNNAAA. The hit rate is lower than that achieved by pytesseract: in a test with 21 images, 11 hits are reached  while with pytesseract the hits are 17 (https://github.com/ablanco1950/LicensePlate_Labeled_MaxFilters).

Requirements:

have the packages installed that allow:

import numpy

import tensorflow

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense, Dropout

import cv2

Download from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ the EnglishImg.tgz file, unzip it and delete the folders that contain the lowercase characters, it is assumed that having fewer classes will make it easier to make predictions and car license plates only admit uppercase.
Rename the downloaded folder as EnglishImgRedu and check that in the path C:\EnglishImgRedu\EnglishImg\English\Img\GoodImg\Bmp there are only the folders from Sample001 to Sample036

In the download directory you should find the downloaded test6Training.zip and must unzip folder: test6Training with all its subfolders, containing the images for the test and its labels. This directory must be in the same directory where is the program GetNumberSpanishLicensePlate_OCRChar64K_labels_MaxFilters.py ( unziping may create two directories with name test6Training and the images may not be founded when executing it, it would be necessary copy of inner directory test6Training in the same directory where is LicensePlateFindContours.py)

Operative:

Create the model using keras CNN, by running:

OCRfromScratchKerasCNN_Chars74k_SpanishLicensePlate.py

Test the model by running:
GetNumberSpanishLicensePlate_OCRChar64K_labels_MaxFilters.py

Each car license plate appears on the screen with the text that could have been recognized from the image and the final result assigning the car license plate that has been recognized the most times.

As output, the LicenseResults.txt file is also obtained with the relation between true license plate and predicted license plate.
