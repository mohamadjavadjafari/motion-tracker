#!/usr/bin/env python
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import imutils


class MotionTracker:
    def __init__(self):
        self.bridge = CvBridge()
        self.min_area = 50
        self.max_area = 50

        self.last_frame = None
        self.kernel = np.ones((5, 5), np.uint8)

        rospy.Subscriber('/usb_cam/image_raw', Image, self._callback)

    def _callback(self, Image):
        image = self.bridge.imgmsg_to_cv2(Image, "bgr8")
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray,
                               100,
                               255,
                               cv2.THRESH_BINARY_INV)[1]

        thresh = cv2.morphologyEx(thresh,
                                  cv2.MORPH_OPEN,
                                  self.kernel)

        if self.last_frame is None:
            self.last_frame = thresh
            return
        frame_delta = cv2.absdiff(self.last_frame, thresh)
        self.last_frame = thresh

        cnts = cv2.findContours(frame_delta,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            if cv2.contourArea(c) < 40:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            solidity = cv2.contourArea(c) / (w * h)
            if solidity < 0.5:
                continue
            if w > 70 and h > 70:
                continue
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("frame_delta", frame_delta)
        cv2.imshow("Motion-Tracker", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node('motion_tracker_node')
    motion = MotionTracker()
    rospy.spin()
