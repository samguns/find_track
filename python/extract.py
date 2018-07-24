import cv2
import os

cap = cv2.VideoCapture('project_video.mp4')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('Error: create directory failed')

currentFrame = 0
while(True):
    _, frame = cap.read()
    name = './data/frame'+str(currentFrame) + '.jpg'
    print('Creating...' + name)
    cv2.imwrite(name, frame)

    currentFrame += 1

cap.release()
cv2.destroyAllWindows()