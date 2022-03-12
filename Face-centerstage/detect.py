import cv2
import mediapipe as mp
import numpy as np
import time
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


'''

Credits: https://github.com/adarsh1021/facedetection 

Referenced to the Video croppping code from thje repository and modified to work with mediapipe.

'''


cap = cv2.VideoCapture(0)

threshold = 300

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.63) as face_detection:
  while cap.isOpened():

    # set timeout to reduce jitter
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    success, image = cap.read()
    height, width = image.shape[:2]
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    RESULTS = []

    if results.detections:
      for detection in results.detections:
        
        x1, y1, w, h = detection.location_data.relative_bounding_box.xmin, detection.location_data.relative_bounding_box.ymin, detection.location_data.relative_bounding_box.width, detection.location_data.relative_bounding_box.height
        x2, y2 = x1 + w, y1 + h
        
        # convert to absolute coordinates from relative
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        w = int(w * width)
        h = int(h * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        if x2 > width:
            x2 = width
        if y2 > height:
            y2 = height
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0

        RESULTS.append((x1, y1, x2, y2))

        # mp_drawing.draw_detection(image, detection)

    if len(RESULTS) < 1:
        x1 = 0
        y1 = 0
        x2 = width
        y2 = height

    
    if len(RESULTS) > 1:
        RESULTS = np.array(RESULTS)
        RESULTS[:, 2] += RESULTS[:, 0]
        RESULTS[:, 3] += RESULTS[:, 1]
        x1 = np.min(RESULTS, axis=0)
        y1 = x1[1]
        x1 = x1[0]
        x2 = np.max(RESULTS, axis=0)
        y2 = x2[3]
        x2 = x2[2]
    
    crp_c0 = x1 - threshold if x1-threshold > 0 else 0
    crp_c1 = x2 + threshold if x2+threshold < width else width
    crp_r0 = y1 - threshold if y1-threshold > 0 else 0
    crp_r1 = y2 + threshold if y2+threshold < height else height

    # rescale with same aspect ratio if crp_c1 - crp_c0 < width
    # rescale with same aspect ratio if crp_r1 - crp_r0 < height    
    aspect_ratio = width / height

    print(crp_c0, crp_c1, crp_r0, crp_r1)

    center_super_width = width / 2
    center_super_height = height / 2
    print("crosshair", center_super_width, center_super_height)
    print("VALUES", crp_c0, crp_c1, crp_r0, crp_r1)

    # centralize the image
    # crp_c0 = int(crp_c0 - center_super_width)
    # crp_c1 = int(crp_c1 - center_super_width)
    # crp_r0 = int(crp_r0 - center_super_height)
    # crp_r1 = int(crp_r1 - center_super_height)


    # Flip the image horizontally for a selfie-view display.    
    cv2.imshow('center stage', cv2.flip(image[crp_r0:crp_r1, crp_c0:crp_c1], 1))
    cv2.imshow('MediaPipe Face Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
