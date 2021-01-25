# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
#CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", 
           #"diningtable","dog", "horse", "motorbike", "person", "pole", "sheep","sofa", "traffic light", "train"]
CLASSES= [line.strip() for line in open('class_labels.txt')]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
# initialize the video stream, allow the camera sensor to warm up,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = cv2.VideoCapture('dashcam_test.mp4')

# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()
w = int(vs.get(3))
h = int(vs.get(4))
out = cv2.VideoWriter('object_distance_output_test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (w,h))
F = 615
count_of_cars = 0
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = vs.read()
    if ret == True: 
        #frame = imutils.resize(frame, width=400)
        # grab the frame dimensions and convert it to a blob
        #(h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        #print(detections)
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if(idx == 7.00):
                    count_of_cars = count_of_cars + 1
                    height = round(endY-startY,4)
                    # Distance from camera based on triangle similarity
                    distance = (165 * F)/height
                    #print("Distance from car(cm):{dist}\n".format(dist=distance))
                    #print(distance)
                    if distance <=500.0:
                        print("Distance from car(cm):{dist}\n".format(dist=distance))
                        #idx = int(detections[0, 0, i, 1])
                        #box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        #(startX, startY, endX, endY) = box.astype("int")
                    label = "{}, distance: {:.2f}cm".format(CLASSES[idx],
                           distance)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                            COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255] , 2)    # draw the prediction on the frame
                    label2 = "distance: {:.2f}%".format(distance)
                    
                    print(count_of_cars)          
        # show the output frame
        cv2.imshow("Frame", frame)
        out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()
    else:
        break
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
out.release()