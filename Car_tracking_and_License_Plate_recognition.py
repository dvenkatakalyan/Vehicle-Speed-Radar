import cv2
import math
import time
import os
import easyocr
import imutils
import numpy as np
import shutil

limit = 80  # km/hr

traffic_record_folder_name = "Traffic"

if not os.path.exists(traffic_record_folder_name):
    os.makedirs(traffic_record_folder_name)
    os.makedirs(traffic_record_folder_name+"//exceeded")


speed_record_file_location = traffic_record_folder_name + "//SpeedRecord.txt"
file = open(speed_record_file_location, "w")
file.write("ID \t SPEED\n------\t-------\n")
file.close()


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        self.id_count = 0
        # self.start = 0
        # self.stop = 0
        self.et = 0
        self.s1 = np.zeros((1, 1000))
        self.s2 = np.zeros((1, 1000))
        self.s = np.zeros((1, 1000))
        self.f = np.zeros(1000)
        self.capf = np.zeros(1000)
        self.count = 0
        self.exceeded = 0

    def update(self, objects_rect):
    #To check if the object is same or not.When an object slightly changes in frame we need to see whether it is same object or not.
    
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # CHECK IF OBJECT IS DETECTED ALREADY
            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
            #How we do that is finding distance between the object in frame 1 and frame 2
            #If the object initially in 8px goes to 7px.Then we say it is same obj.
                if dist < 70: #This is the distance between 2 of them
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
            #When I used single integer to calculate time difference,it will not consider if there are 3 vehicles on the same segment.
            #Very difficult to see which car starts the time and which car stops the timer.
            #Thats why created an array with ID's.
            #ID 1 car has its own start and stop timer.
            #ID 2 car has its own start and stop timer.
                    # START TIMER when vehicle crosses first line
                    if (y >= 410 and y <= 430):
                        self.s1[0, id] = time.time()

                    # STOP TIMER when vehicle crosses second line and FIND DIFFERENCE
                    if (y >= 235 and y <= 255):
                        self.s2[0, id] = time.time()
                        self.s[0, id] = self.s2[0, id] - self.s1[0, id]

                    # CAPTURE FLAG
                    #when vehicle crosses last line and speed is estimated then flag is 1 
                    #when flag=1 it captures the image.
                    #Once the image is cature and saved the flag would be zero.
                    if (y < 235):
                        self.f[id] = 1

            # NEW OBJECT DETECTION
            #In case the distance between object in frame 1 and frame 2 is not lesser than 70 a new object would be detected and new id will be given to it.
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.s[0, self.id_count] = 0
                self.s1[0, self.id_count] = 0
                self.s2[0, self.id_count] = 0

        # ASSIGN NEW ID to OBJECT
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    # SPEEED FUNCTION:Find speed of vehicle
    def getsp(self, id):
        if (self.s[0, id] != 0):
            s = 214.15 / self.s[0, id]
            #The easiest way to calculate the number(214.15)(distance) is by taking a reference vehicle first, and calculate the speed backwards. (which I used) 
            #A more complicated approach would be to take into consideration the distance between the lines, time taken, frame rate and lag due to computation. 
        else:
            s = 0

        return int(s)

    # SAVE VEHICLE DATA
    #Crops image and saves
    def capture(self, img, x, y, h, w, sp, id):
        if (self.capf[id] == 0):
            self.capf[id] = 1
            self.f[id] = 0
            crop_img = img[y - 5:y + h + 5, x - 5:x + w + 5]
            #n = str(id) + "_speed_" + str(sp)
            n="Cars"+str(id)
            file = traffic_record_folder_name + '//' +'cropped_images_of_cars' +'//' + n + '.jpg'
            cv2.imwrite(file, crop_img)
            self.count += 1
            filet = open(speed_record_file_location, "a")
            if (sp > limit): #Vehicle crosses speed limit
                file2 = traffic_record_folder_name + '//exceeded//' + '//Overspeeding_cars//' + n + '.jpg'
                cv2.imwrite(file2, crop_img)
                filet.write(str(id) + " \t " + str(sp) + "<---exceeded\n")
                self.exceeded += 1
            else:
                filet.write(str(id) + " \t " + str(sp) + "\n")
            filet.close()

    # SPEED_LIMIT:just returns the speed limit mentioned earlier.
    def limit(self):
        return limit


    def read_license_plates(self,folder_path):
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])
        location = None
        count=0
        # Get a list of image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        with open (traffic_record_folder_name + '/' + '/exceeded/Numberplates/Automatically_detected/'+'/Number_plates.txt', 'w') as file:
            # Loop through each image file
            for image_file in image_files:
                count=count+1
                # Read the license plate number using EasyOCR
                image_path = os.path.join(folder_path, image_file)
                #print("image_path",image_path)
                k=cv2.imread(image_path)
                gray = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)
                bfilter = cv2.bilateralFilter(gray, 11, 11, 17)
                edged = cv2.Canny(bfilter, 30, 200)
                keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = imutils.grab_contours(keypoints)
                contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
                for contour in contours:
                    approx = cv2.approxPolyDP(contour, 10, True)
                    if len(approx) == 4:
                        location = approx
                        break
                mask = np.zeros(gray.shape, np.uint8)
                new_image = cv2.drawContours(mask, [location], 0, 255, -1)
                new_image = cv2.bitwise_and(k,k, mask = mask)
                (x, y) = np.where(mask == 255)
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_image = gray[x1:x2+3, y1:y2+3]
                reader = easyocr.Reader(['en'])
                result = reader.readtext(cropped_image)
                if result!=[]:
                    text = result[0][1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    res = cv2.putText(k, text = text, org = (approx[0][0][0], approx[1][0][1]+60), fontFace = font, fontScale = 1, color = (0, 255, 0), thickness = 5)
                    res = cv2.rectangle(k, tuple(approx[0][0]), tuple(approx[2][0]), (0,255, 0), 3)       
                    filename=traffic_record_folder_name+'/'+  f'exceeded/Numberplates/Automatically_detected/' + f'car{count}.jpg'
                    cv2.imwrite(filename, res)
                    file.write(result[0][1])
                    file.write('\n') 
                else:
                    src=image_path
                    dst=traffic_record_folder_name + '/' + '/exceeded/Numberplates/Manual_inspection_needed/' + image_file
                    shutil.copy(src, dst)
                    #continue
            # Wait for 1 second before checking for new images
            time.sleep(1)



    # TEXT FILE SUMMARY
    def end(self):
        file = open(speed_record_file_location, "a")
        file.write("\n-------------\n")
        file.write("-------------\n")
        file.write("SUMMARY\n")
        file.write("-------------\n")
        file.write("Total Vehicles :\t" + str(self.count) + "\n")
        file.write("Exceeded speed limit :\t" + str(self.exceeded))
        file.close()
