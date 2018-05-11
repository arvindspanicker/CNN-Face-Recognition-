import cv2
import dlib
from imutils import face_utils

camera = cv2.VideoCapture('rtmp://192.168.20.144:1935/flash/12:admin:EYE3inapp')


while True:
	ret, img = camera.read()
	img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(400,400))
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	rects = detector(img, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		for (x, y) in shape:
			cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
	cv2.imshow("Frame", img)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
