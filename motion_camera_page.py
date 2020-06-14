from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import RPi.GPIO as GPIO


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.IN)         #Read output from PIR motion sensor


def take_picture_face():
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
	net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

	# initialize the video stream, allow the cammera sensor to warmup,
	# and initialize the FPS counter
	vs = VideoStream(src=0).start()
	# vs = VideoStream(usePiCamera=True).start()
	time.sleep(2.0)
	fps = FPS().start()

	# loop over the frames from the video stream
	
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
			if idx==15:
				label = "person"
				cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				cv2.imwrite('image.jpg',frame)
	vs.stop()
def servo_gate_close():
    p.ChangeDutyCycle(12.5) # turn towards 180 degree
    time.sleep(1) # sleep 1 second
def servo_gate_close():
    p.ChangeDutyCycle(2.5) # turn towards 0 degree
    time.sleep(1) # sleep 1 second
def read_data_iot():
	with urllib.request.urlopen(
                f"https://api.thingspeak.com/channels/670394/fields/1/last.json??timezone=Asia%2FKolkata") as url:
            data = json.loads(url.read().decode())
            return data["feeds"] 



GPIO.setup(12, GPIO.OUT)# servo pin 12

p = GPIO.PWM(12, 50)

p.start(7.5)
while True:
	i=GPIO.input(11)			# pir pin 11
	if i==0:                 	#When output from motion sensor is LOW, No motion
		pass
	elif i==1:
		e=read_data_iot()[0]['created_at'][-3:-1] 

		take_picture_face() #upload picture
		
		q=read_data_iot()[0]
		if q['created_at'][-3:-1]!=e:
			if q['field1']==str(2): 
				servo_gate_open()
				time.sleep(5)
				servo_gate_close()
			elif q['field1']==str(3): 
				servo_gate_close()
				
