#!/usr/bin/env python

import cv2
import numpy as np
from opencv_rect import CvRect
from motion_tracker import MotionTracker
from timeit import default_timer as timer
from subprocess import Popen

def image_roi(img, roi_CvRect):
    return img[ roi_CvRect.x:roi_CvRect.x + roi_CvRect.w,
                roi_CvRect.y:roi_CvRect.y + roi_CvRect.h]

class ObjectTracker(MotionTracker):

    def __init__(self, videosource, objcet_cascade, filter_alpha = None, motion_threshold = None):
        super(ObjectTracker, self).__init__(videosource, filter_alpha, motion_threshold)

        self.obj_classifier = cv2.CascadeClassifier(objcet_cascade)

    def detect_object(self):

        if (self.prev_frame is not None and
            self.motion_roi.area() > 0 and
            not self.obj_classifier.empty()):

            objects = self.obj_classifier.detectMultiScale(image_roi(self.prev_frame, self.motion_roi))

            if len(objects) > 0:
                obj_rects = list()
                for o in objects:
                    obj_rects.append(CvRect(o))

                return obj_rects
            return None

def test_object_tracker():

    mt = ObjectTracker(0, "../data/fist.xml")

    cv2.namedWindow("Motion", cv2.WINDOW_NORMAL)

    time_of_detect_start = None
    is_detected = False
    for frame in mt.all_input_frames():

        mt.process_frame(frame)

        if mt.motion_roi is not None:
            objs = mt.detect_object()
        
            if objs is not None:
                num_objs = len(objs)

                if num_objs == 1 and time_of_detect_start is None:
                    time_of_detect_start = timer()
                    print("Start")
                elif num_objs != 1:
                    print("Reset")
                    time_of_detect_start = None

                prev_detected = is_detected
                now_time = timer()
                if (time_of_detect_start is not None and
                    (now_time - time_of_detect_start) > 1.5):
                    is_detected = True
                elif time_of_detect_start is None:
                    is_detected = False

                if is_detected and not prev_detected:
                    Popen("uxterm")

                for obj_rect in objs:
                    cv2.rectangle(image_roi(frame, mt.motion_roi), obj_rect.tl(), obj_rect.br(), (100, 255, 0), 2)

            cv2.rectangle(frame, mt.motion_roi.tl(), mt.motion_roi.br(), (255, 100, 0), 2)

        cv2.imshow("Motion", frame)
        if cv2.waitKey(1) & 0xFF == 27:
           break

    mt.clean_up()

if __name__ == "__main__":
    test_object_tracker()