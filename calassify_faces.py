# check opencv version
import cv2
import mtcnn

pixels = cv2.imread('images/test1.jpg')

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    detector = mtcnn.MTCNN()
    faces = detector.detect_faces(frame)
    bboxes = classifier.detectMultiScale(frame)
    for box in faces:
        print(box)
        x, y, width, height = box.get("box")
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
    cv2.imshow('face detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# show the image
# keep the window open until we press a key
cv2.waitKey(0)
# close the window
cv2.destroyAllWindows()