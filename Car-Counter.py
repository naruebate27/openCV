import numpy as np
from ultralytics import YOLO
import cv2
import mysql.connector
from mysql.connector import errorcode
from vidgear.gears import CamGear
import cvzone
import calendar
from datetime import datetime
import time
import math
from sort import *

mydb = mysql.connector.connect(
  host='localhost',
  user='root',
  password='',
  database='detectdb'
)
if mydb.is_connected():
    db_Info = mydb.get_server_info()
    print("Connected to MySQL Server version ", db_Info)
    cursor = mydb.cursor()
options = {"CAP_PROP_FRAME_WIDTH":1920, "CAP_PROP_FRAME_HEIGHT":1080}
cap = CamGear(source='https://www.youtube.com/watch?v=En_3pkxIJRM', stream_mode=True, logging=True, **options).start()
# cap = cv2.VideoCapture('cars.mp4')  # For Video
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(width,height)
model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask2.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# limits = [400, 297, 373, 297]
limit1 = [750,780,1267,565]
limit2 = [739,804,1110,1070]
totalCarCount = []
totalTruckCount = []
totalBusCount = []
totalMotorbikeCount = []
totalPersonCount = []

while True:
    img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgRegion, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]



            if currentClass == "car" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

                resultsTracker = tracker.update(detections)

                cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (0, 0, 255), 3)
                for result in resultsTracker:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=2, offset=5)

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

                    if limit1[0] < cx < limit1[2] and limit1[1] - 15 < cy < limit1[1] + 15:
                        if totalCarCount.count(id) == 0:
                            totalCarCount.append(id)
                            cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (0, 255, 0), 2)

            elif currentClass == "truck" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                resultsTracker2 = tracker.update(detections)

                for result in resultsTracker2:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=2, offset=5)

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

                    if limit1[0] < cx < limit1[2] and limit1[1] - 15 < cy < limit1[1] + 15:
                        if totalTruckCount.count(id) == 0:
                            totalTruckCount.append(id)
                            cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (0, 255, 0), 2)

            elif currentClass == "bus" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                resultsTracker3 = tracker.update(detections)

                for result in resultsTracker3:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=2, offset=5)

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

                    if limit1[0] < cx < limit1[2] and limit1[1] - 15 < cy < limit1[1] + 15:
                        if totalBusCount.count(id) == 0:
                            totalBusCount.append(id)
                            cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (0, 255, 0), 2)

            elif currentClass == "motorbike" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                resultsTracker4 = tracker.update(detections)

                for result in resultsTracker4:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=2, offset=5)

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

                    if limit1[0] < cx < limit1[2] and limit1[1] - 15 < cy < limit1[1] + 15:
                        if totalMotorbikeCount.count(id) == 0:
                            totalMotorbikeCount.append(id)
                            cv2.line(img, (limit1[0], limit1[1]), (limit1[2], limit1[3]), (0, 255, 0), 2)

            elif currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                resultsTracker5 = tracker.update(detections)

                cv2.line(img, (limit2[0], limit2[1]), (limit2[2], limit2[3]), (0, 0, 255), 3)

                for result in resultsTracker5:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=8, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=2, offset=5)

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

                    if limit2[0] < cx < limit2[2] and limit2[1] - 15 < cy < limit2[1] + 15:
                        if totalPersonCount.count(id) == 0:
                            totalPersonCount.append(id)
                            cv2.line(img, (limit2[0], limit2[1]), (limit2[2], limit2[3]), (0, 255, 0), 2)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, "cars detected : "+ str(len(totalCarCount)),(55,50),cv2.FONT_HERSHEY_PLAIN,2,(50,50,255),2)
    cv2.putText(img, "trucks detected : "+ str(len(totalTruckCount)),(55,100),cv2.FONT_HERSHEY_PLAIN,2,(50,50,255),2)
    cv2.putText(img, "Buses detected : "+ str(len(totalBusCount)),(55,150),cv2.FONT_HERSHEY_PLAIN,2,(50,50,255),2)
    cv2.putText(img, "Motorbike detected : "+ str(len(totalMotorbikeCount)),(55,200),cv2.FONT_HERSHEY_PLAIN,2,(50,50,255),2)
    cv2.putText(img, "Persons detected : "+ str(len(totalPersonCount)),(55,250),cv2.FONT_HERSHEY_PLAIN,2,(50,50,255),2)

    Car = len(totalCarCount)
    Truck = len(totalTruckCount)
    Bus = len(totalBusCount)
    Motorbike = len(totalMotorbikeCount)
    Person = len(totalPersonCount)
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    date_time = datetime.fromtimestamp(time_stamp)
    Date_time = date_time
    print(Car, Person, Date_time)
    add_dta = """INSERT INTO detect_tb (car, truck, bus, motorbike, person, time ) VALUES ( %s, %s, %s, %s, %s, %s  )"""

    cursor.execute(add_dta, (Car, Truck, Bus, Motorbike, Person, Date_time))
    mydb.commit()

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)