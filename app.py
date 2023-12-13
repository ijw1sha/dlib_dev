lass VideoProcessor:


    def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=False):
        points = []

        for i in range(startpoint, endpoint+1):
            point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
            points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
        	s = predictor(gray, rect)
        	s = face_utils.shape_to_np(s)

            s_ = predictor(gray, rect)
        	s = face_utils.shape_to_np(s_)

        	# for(i, y) in s:
        	# 	cv2.circle(img, (i,y), 2, (0, 255, 0), -1)

        	for(i, y) in s:
        		cv2.circle(img, (i,y), 2, (0, 255, 0), -1)
            # count number of landmarks we actually detected on image

            if i==0:
                print("Total number of face landmarks detected ",len(s_.parts()))

                drawPoints(img, s_, 0, 16)           # Jaw line
                drawPoints(img, s_, 17, 21)          # Left eyebrow
                drawPoints(img, s_, 22, 26)          # Right eyebrow
                drawPoints(img, s_, 27, 30)          # Nose bridge
                drawPoints(img, s_, 30, 35, True)    # Lower nose
                drawPoints(img, s_, 36, 41, True)    # Left eye
                drawPoints(img, s_, 42, 47, True)    # Right Eye
                drawPoints(img, s_, 48, 59, True)    # Outer lip
                drawPoints(img, s_, 60, 67, True)    # Inner lip 

        return av.VideoFrame.from_ndarray(img, format="bgr24")
