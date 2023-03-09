
# Vehicle Speed Radar

This project mainly focuses on Tracking of all cars from CCTV surveillance camera and extracting license plate of the overspeeding cars using the techniques of Image processing. 


# Acknowledgements

 - [Vehicle speed measurement model for video-based systems](https://doi.org/10.1016/j.compeleceng.2019.04.001s)
 - [Fast and accurate on-road vehicle detection based on color intensity segregation](https://doi.org/10.1016/j.procs.2018.07.090)
 - [Determining vehicle speed based on video using convolutional neural network](https://doi.org/10.1016/j.trpro.2020.10.024)


# Flow Chart 

![Alt text](https://drive.google.com/file/d/1yPnDwa_Q9AuERsMdtxY-5hcvAlXZ3up-/view?usp=sharing)
# Appendix

### SpeedRadar
This code mainly focuses on Tracking of vehicles using background subtraction.

### Car_tracking_and_License_Plate_recognition
This code mainly focuses on assigning ID to the vehicles and License plate number extraction using EasyOCR.

### Video
Any Similar Video would do, however there would be some changes you'll have to make to the code if you choose a different video. 

This is due to a change of scene, distance estimation and pixel clarity. 

For SpeedRadar.py, the region of interest, the red timer lines and the area threshold used for detecting vehicles.

For Car_tracking_and_License_Plate_recognition.py, the numbers given in the update() function and getsp() fuction would change for a different video.

### Files 
Traffic folder consists of folder named cropped_images_of_cars where I have stored all the cropped images of car that are being tracked using background subtraction.

There is another folder named exceeded which consists of all the cars that have exceeded the speed limit. 

The easyOCR library sometimes failes to extract the Number plate of the overspeeding vehicle, So I have made 2 folders in Numberplates(Automatically_detected,Manual_inspection_needed).

Automatically_detected folder contains the licence plate numbers of overspeeding cars that are automatically extracted by easyOCR.

Manual_inspection_needed folder contains the licence plate numbers of overspeeding cars that are to be manually inspected by a person as easyOCR fails to extract licence plate number.


## Deployment

To deploy this project run
Create a virtual environment and install all the required libraries using 
```bash
  pip install -r requirements.txt
```

To deploy this project run
```bash
  python SeedRadar.py
```
## Demo

Insert gif or link to demo

