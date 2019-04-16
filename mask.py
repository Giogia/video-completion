import cv2
import numpy as np
from PIL import Image
from math import pi, cos, sin

video = cv2.VideoCapture('demo.mp4')

# Check if camera opened successfully
if not video.isOpened():
    print("Error opening video stream or file")

# Get clip information
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video
out = cv2.VideoWriter('output.mp4', 0x7634706d, fps, (width,height))

# Create mask to apply
mask = Image.fromarray(255 * np.ones((height, width), np.uint8))

# Center offset
x = width / 2
y = height / 2

# Initialize parameters
X = x
Y = y
S = 1
DEG = 0
THRESHOLD = 10
BOUNCING = 2

# Read until video is completed
while video.isOpened():

    # Capture frame-by-frame
    ret, frame = video.read()
    if ret:

        # Rotate mask of a pseudo random angle RAD
        if DEG < -THRESHOLD:
            DEG += 5 * np.random.uniform(-0.5, 2)
        if DEG > THRESHOLD:
            DEG -= 5 * np.random.uniform(-0.5, 2)
        else:
            DEG += np.random.uniform(-1, 1)

        RAD = DEG * pi / 180

        # Translate mask horizontally of a pseudo random value X
        if X < 0.9 * x:
            X += 50 * np.random.uniform(-0.5, 2)
        if X > 1.1 * x:
            X -= 50 * np.random.uniform(-0.5, 2)
        else:
            X += 10 * np.random.uniform(-1, 1)

        # Translate mask vertically of a pseudo random value Y
        if Y < 0.9 * y:
            Y += 50 * np.random.uniform(-0.5, 2)
        if Y > 1.1 * y:
            Y -= 50 * np.random.uniform(-0.5, 2)
        else:
            Y += 10 * np.random.uniform(-1, 1)


        params = [cos(RAD), -sin(RAD), X - x * cos(RAD) + y * sin(RAD),
                  sin(RAD), cos(RAD), Y - x * sin(RAD) - y * cos(RAD),
                  0, 0]

        rotated_mask = mask.transform((width, height), Image.PERSPECTIVE, params)
        rotated_mask = np.array(rotated_mask)

        # Apply mask to frame
        frame = cv2.bitwise_and(frame, frame, mask=rotated_mask)
        out.write(frame)

    # Break the loop
    else:
        break

# When everything done, release the video capture object
video.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()