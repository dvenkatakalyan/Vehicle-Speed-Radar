import cv2
from Car_tracking_and_License_Plate_recognition import *
import numpy as np
end = 0

traffic_record_folder_name = "Traffic"

if not os.path.exists(traffic_record_folder_name):
    os.makedirs(traffic_record_folder_name)
    os.makedirs(traffic_record_folder_name+"//exceeded")
# # #Creater Tracker Object
tracker = EuclideanDistTracker()

#cap = cv2.VideoCapture("Resources/traffic3.mp4")
cap = cv2.VideoCapture("cctv_surveillance.mp4")
f = 25
w = int(1000/(f-1))


#Object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(varThreshold=None)
#100,5

#KERNALS for masking technique
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)#Finds the difference between Foreground and background

kernal_e = np.ones((5,5),np.uint8)

while True:
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)#Resizing frame to half of its original size
    height,width,_ = frame.shape
    #print(height,width)
    #540,960


    #Extract ROI:
    roi = frame[50:540,200:960] #Cropping image to only area where there is road and vehicles.

    #MASKING METHOD 1
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

    #DIFFERENT MASKING METHOD 2 -> This is used
    fgmask = fgbg.apply(roi) #Getting foreground mask(fgmask) using apply method in fgbg
    ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)#Thresholding to create binary image
    #Morphological operation
    #Generally morphological operations done on binary image. We do thresholding to get that binary image.
    mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)#opening used to remove noise
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernalCl)#opening used to remove noise
    e_img = cv2.erode(mask2, kernal_e) #Erosion:pixels are removed at the boundaries. Features are better highlighted.
                                        #Small noise or unwanted details removed
    #Here I am directly using eroded image for detection instead I can also use bitwise_and function with mask to refine image and remove noise.
    #Using the eroded image directly can be useful in cases where you want to speed up the processing time or simplify the code
    contours,_ = cv2.findContours(e_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#used to find the shape of objects that are detected and saves the details of where it is and how big the image is.
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #THRESHOLD for finding vehicles
        if area > 1000:#If area of contour>1000 then it is a car 
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detections.append([x,y,w,h])

    #Object Tracking
    boxes_ids = tracker.update(detections) #assigning Id to bounding boxes
    for box_id in boxes_ids:
        x,y,w,h,id = box_id


        if(tracker.getsp(id)<tracker.limit()):#Speed is less than the limit put a green rectangle
            cv2.putText(roi,str(id)+" "+str(tracker.getsp(id)),(x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        else:#Speed is more than the limit put a orange rectangle
            cv2.putText(roi,str(id)+ " "+str(tracker.getsp(id)),(x, y-15),cv2.FONT_HERSHEY_PLAIN, 1,(0, 0, 255),2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 165, 255), 3)

        s = tracker.getsp(id)
        if (tracker.f[id] == 1 and s != 0):
            tracker.capture(roi, x, y, h, w, s, id) #save image

    # DRAW LINES
    #Reason for 4 lines instead of 2 lines. Sometimes when vehicle passes over a particular line it did not sense it.So needed to use small segment of road to start timer and end timer.
    cv2.line(roi, (0, 410), (960, 410), (0, 0, 255), 2)
    cv2.line(roi, (0, 430), (960, 430), (0, 0, 255), 2)

    cv2.line(roi, (0, 235), (960, 235), (0, 0, 255), 2)
    cv2.line(roi, (0, 255), (960, 255), (0, 0, 255), 2)


    #DISPLAY
    #cv2.imshow("Mask",mask2)
    #cv2.imshow("Erode", e_img)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(w-10)
    if key==27:
        tracker.end() #Used to create text files
        end=1
        break

if(end!=1):
    tracker.end()

cap.release()
cv2.destroyAllWindows()
folder_path=traffic_record_folder_name+ "//" + "exceeded" +  "//" + "Overspeeding_cars"
tracker.read_license_plates(folder_path)
