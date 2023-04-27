import cv2

cap  = cv2.VideoCapture("los_angeles.mp4")
ret, frame = cap.read()
ret, frame1 = cap.read()
ret, frame2 = cap.read()

#object detection from stable camera
#object_detetcor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30)


while cap.isOpened():

    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    _,thresh=cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours,_ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


 #   ret, frame = cap.read()
 #   height, width, _ = frame.shape
    #print(height, width)

    #extracting region of interest
  #  roi = frame[450:1080,450:1920]

    # object detection
  #  mask = object_detetcor.apply(roi)
  #  _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
  #  contours, _ =  cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        #calculating area and removing small elements
        #area = cv2.contourArea(cnt)
        #if area > 600:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
        (x, y, w, h) = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < 700:
            continue
        cv2.rectangle(frame1, (x,y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    key = cv2.waitKey(30)
    if key == 27:
       break

cap.release()  
cv2.destroyAllWindows()

