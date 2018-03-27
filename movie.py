import numpy as np
import cv2
import copy
import time

class Topview:
    width=640
    height=480

    def __init__(self, front, back, left, right):
        self.front = front
        self.back = back
        self.left = left
        self.right = right

    def capture(self):
        self.__init_capture()
        self.__make_mask()

        while(True):
            start = time.time()

            res = self.__get_screen()

            front = self.__capture_front()
            back = self.__capture_back()
            left = self.__capture_left()
            right = self.__capture_right()

            if isinstance(front, np.ndarray):
                res = cv2.bitwise_or(res, front)
            if isinstance(back, np.ndarray):
                res = cv2.bitwise_or(res, back)
            if isinstance(left, np.ndarray):
                res = cv2.bitwise_or(res, left)
            if isinstance(right, np.ndarray):
                res = cv2.bitwise_or(res, right)

            # Show combined image
            cv2.imshow('frame', res)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if (
                not(isinstance(front, np.ndarray)) and
                not(isinstance(back, np.ndarray)) and
                not(isinstance(left, np.ndarray)) and
                not(isinstance(right, np.ndarray))
            ):
                break

            # ~ 24 frames per second
            if time.time() - start < 0.04:
                time.sleep(0.04 - (time.time() - start))
        
        # Release everything if job is finished
        self.frontCapture.release()
        self.backCapture.release()
        self.leftCapture.release()
        self.rightCapture.release()
        cv2.destroyAllWindows()

    def __init_capture(self):
        self.frontCapture = cv2.VideoCapture(self.front)
        self.backCapture = cv2.VideoCapture(self.back)
        self.leftCapture = cv2.VideoCapture(self.left)
        self.rightCapture = cv2.VideoCapture(self.right)

    def __get_screen(self):
        return np.zeros(
                (
                    self.height*2 + int(self.width/2),
                    self.height*2 + int(self.width/2),
                    3
                ),
                np.uint8
            )

    def __make_mask(self):
        self.mask = np.zeros((self.height, self.width, 3), np.uint8)
        # Draw a white, filled rectangle on the mask image
        pts = np.array(
            [
                [0, 0],
                [self.width, 0],
                [int(self.width*3/4), self.height],
                [int(self.width/4), self.height]
            ],
            np.int32
        )
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(self.mask, [pts], (255, 255, 255))

    def __apply_mask(self, frame):
        # Resize image to fit self.width and self.height
        frame = cv2.resize(frame, (self.width, self.height), fx=0, fy=0)

        # Apply the mask
        frame = cv2.bitwise_and(frame, self.mask)
        return frame

    def __change_perspective(self, frame):
        # FIXME Do not change perspective for now, until real data is not provided
        return frame
        # Change perspective
        pts1 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
        pts2 = np.float32([[0, 0], [self.width, 0], [0, int(self.width/4)], [self.width, int(self.width/4)]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        frame = cv2.warpPerspective(frame, M, (self.width, self.height))
        return frame

    def __capture_front(self):
        if not(self.frontCapture.isOpened):
            return None

        ret, frame = self.frontCapture.read()
        if ret == True:
            frame = self.__apply_mask(frame)
            frame = self.__change_perspective(frame)

            #create empty matrix
            vis = self.__get_screen()

            # TOP
            vis[
                :self.height,
                self.height-int(self.width/4):self.height-int(self.width/4)+self.width,
                :3
            ] = frame
            return vis
        
        return None

    def __capture_back(self):
        if not(self.frontCapture.isOpened):
            return None

        ret, frame = self.frontCapture.read()
        if ret == True:
            frame = self.__apply_mask(frame)
            frame = self.__change_perspective(frame)

            #create empty matrix
            vis = self.__get_screen()

            # BACK
            vis[
                self.height+int(self.width/2):self.height*2 + int(self.width/2),
                self.height-int(self.width/4):self.height-int(self.width/4)+self.width,
                :3
            ] = cv2.flip(frame, -1)
            return vis
        
        return None

    def __capture_left(self):
        if not(self.frontCapture.isOpened):
            return None

        ret, frame = self.frontCapture.read()
        if ret == True:
            frame = self.__apply_mask(frame)
            frame = self.__change_perspective(frame)

            #create empty matrix
            vis = self.__get_screen()

            # LEFT
            M = cv2.getRotationMatrix2D((self.width/2, self.width/2), 90, 1)
            frame = cv2.warpAffine(frame, M, (self.width, self.width))
            frame = frame[0:self.width, 0:self.height]
            vis[
                self.height-int(self.width/4):self.width+self.height-int(self.width/4),
                :self.height,
                :3
            ] = frame
            return vis
        
        return None

    def __capture_right(self):
        if not(self.frontCapture.isOpened):
            return None

        ret, frame = self.frontCapture.read()
        if ret == True:
            frame = self.__apply_mask(frame)
            frame = self.__change_perspective(frame)

            #create empty matrix
            vis = self.__get_screen()

            # RIGHT
            M = cv2.getRotationMatrix2D((self.width/2, self.width/2), -90, 1)
            frame = cv2.warpAffine(frame, M, (self.width, self.width))
            frame = frame[0:self.width, self.width-self.height:self.width]
            vis[
                self.height-int(self.width/4):self.width+self.height-int(self.width/4),
                self.height + int(self.width/2):self.height*2 + int(self.width/2),
                :3
            ] = frame
            return vis
        
        return None

topview = Topview('video.mp4', 'video.mp4', 'video.mp4', 'video.mp4')
topview.capture()