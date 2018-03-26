import numpy as np
import cv2
import copy
import time

# cap = cv2.VideoCapture('http://api.new.livestream.com/accounts/7970204/events/2926299/live.m3u8')
cap = cv2.VideoCapture('video.mp4')

if cap.isOpened():
    ret, frame = cap.read()
    if ret==True:
        mask = np.zeros(frame.shape, dtype = "uint8")
        # Draw a white, filled rectangle on the mask image
        oh, ow = frame.shape[:2]
        pts = np.array([[0,0],[ow,0],[int(ow*3/4),oh],[int(ow/4),oh]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(mask, [pts], (255, 255, 255))


        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                start = time.time()
                # FIXME this is for visual purpose only
                frame = cv2.bitwise_not(frame)
                # Apply the mask
                maskedImg = cv2.bitwise_and(frame, mask)
                # Resize image to half of it
                maskedImg = cv2.resize(maskedImg, (0,0), fx=0.5, fy=0.5)
                h, w = maskedImg.shape[:2]

                # Change perspective
                pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
                pts2 = np.float32([[0,0],[w,0],[0,int(w/4)],[w,int(w/4)]])

                M = cv2.getPerspectiveTransform(pts1,pts2)
                maskedImg = cv2.warpPerspective(maskedImg,M,(w,int(w/4)))

                h, w = maskedImg.shape[:2]
                
                #create empty matrix
                visTB = np.zeros((h*2 + int(w/2), h*2 + int(w/2), 3), np.uint8)
                visLR = copy.copy(visTB)

                #combine 4 images
                # TOP
                visTB[:h, h-int(w/4):h-int(w/4)+w, :3] = maskedImg
                # BOTTOM
                visTB[h+int(w/2):h*2 + int(w/2), h-int(w/4):h-int(w/4)+w, :3] = cv2.flip(maskedImg, -1)

                # LEFT
                M = cv2.getRotationMatrix2D((w/2, w/2), 90, 1)
                left = cv2.warpAffine(maskedImg, M, (w, w))
                left = left[0:w, 0:h]
                visLR[h-int(w/4):w+h-int(w/4), :h, :3] = left
                # RIGHT
                M = cv2.getRotationMatrix2D((w/2, w/2), -90, 1)
                right = cv2.warpAffine(maskedImg, M, (w, w))
                right = right[0:w, w-h:w]
                visLR[h-int(w/4):w+h-int(w/4), h + int(w/2):h*2 + int(w/2), :3] = right
                
                cv2.imshow('frame', cv2.bitwise_or(visLR, visTB))
                # print('END', time.time() - start)
                
                # ~ 24 frames per second
                time.sleep(0.04 - (time.time() - start))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()