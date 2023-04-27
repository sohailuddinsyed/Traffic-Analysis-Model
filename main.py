import cv2

cap  = cv2.VideoCapture("los_angeles.mp4")

# object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

# initialize count
count  = 0
center_points = []

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # extracting region of interest
    roi = frame[450:1080,450:1920]

    # object detection and tracking
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ =  cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
         
        # calculating area and removing small elements
        area = cv2.contourArea(cnt)
        if area > 900:
            #  cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)

            # obtaining center of rectangle frames
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points.append((cx,cy))

            cv2.rectangle(roi, (x,y), (x + w, y + h), (0, 255, 0), 2)

            # tracking objects via multiple frames
    for pt in center_points:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
       break

cap.release()  
cv2.destroyAllWindows()

