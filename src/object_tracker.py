#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from opencv_rect import CvRect
from motion_tracker import MotionTracker
from timeit import default_timer as timer
from subprocess import Popen
from PIL import Image

def image_roi(img, roi_CvRect):
    return img[ roi_CvRect.x:roi_CvRect.x + roi_CvRect.w,
                roi_CvRect.y:roi_CvRect.y + roi_CvRect.h]

class ObjectTracker(MotionTracker):

    def __init__(self, videosource, objcet_cascade, filter_alpha = None, motion_threshold = None):
        super(ObjectTracker, self).__init__(videosource, filter_alpha, motion_threshold)

        self.obj_classifier = cv2.CascadeClassifier(objcet_cascade)

    def detectAllObjects(self):

        if (self.prev_frame is not None and
            self.motion_roi.area() > 0 and
            not self.obj_classifier.empty()):

            m_roi = image_roi(self.prev_frame, self.motion_roi)

            try:
                objects = self.obj_classifier.detectMultiScale(m_roi)
            except:
                objects = []

            if len(objects) > 0:
                obj_rects = list()
                for o in objects:
                    obj_rects.append(CvRect(o))

                return obj_rects
            return None

def test_object_tracker():

    tracking_methods = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW" ]
    cur_tracking_method = 0


    mt = ObjectTracker(0, "../data/fist.xml", 0.3)

    cv2.namedWindow("Motion", cv2.WINDOW_NORMAL)

    time_of_detect_start = None
    is_detected = False
    tracker = None
    obj_rect = None
    last_fps = 0
    for frame in mt.all_input_frames():

        if last_fps != mt.last_fps:
                print("FPS: {0}".format(mt.last_fps))
                last_fps = mt.last_fps

        if is_detected:
            if tracker is None:
                print("Initializing tracker, method: {0}, rect: {1}".format(tracking_methods[cur_tracking_method], obj_rect))
                tracker = cv2.Tracker_create(tracking_methods[cur_tracking_method])
                tracker.init(frame, obj_rect)
                prev_centers = [ (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                                 (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                                 (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                                 (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                                 (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),]

            ok, obj_rect = tracker.update(frame)
            rectbbox = CvRect(obj_rect)
            prev_centers.pop(0)
            prev_centers.append(rectbbox.center())

            in_frame_area = rectbbox.intersect(CvRect( (0, 0, frame.shape[1], frame.shape[0])) ).area()


            if ok and in_frame_area >= rectbbox.area():
                for i in range(1, len(prev_centers)):
                    cv2.line(frame, prev_centers[i-1], prev_centers[i], (100, 255, 255), 3)

                cv2.rectangle(frame, rectbbox.tl(), rectbbox.br(), (0, 100, 255), 2)
            else:
                print(ok, in_frame_area >= rectbbox.area())
                is_detected = False
                tracker = None
                print("")


        else:
            mt.process_frame(frame)

            if mt.motion_roi is not None:
                objs = mt.detectAllObjects()
            
                if objs is not None:
                    num_objs = len(objs)
                    sorted_rects = sorted(objs, reverse = True)

                    if num_objs >= 1 and time_of_detect_start is None:
                        time_of_detect_start = timer()
                        print("Start {0}".format(num_objs))
                    elif num_objs == 0:
                        print("Reset {0}".format(num_objs))
                        time_of_detect_start = None

                    prev_detected = is_detected
                    now_time = timer()
                    if (time_of_detect_start is not None and
                        (now_time - time_of_detect_start) > 1.5):
                        is_detected = True
                    elif time_of_detect_start is None:
                        is_detected = False

                    if is_detected and not prev_detected:
                        obj_rect = tuple(sorted_rects[0].shifted(mt.motion_roi.x, mt.motion_roi.y))
                        # Popen(["xdg-open", "../data/success-kid.jpg"])

                    if num_objs >= 1:
                        cv2.rectangle(image_roi(frame, mt.motion_roi), sorted_rects[0].tl(), sorted_rects[0].br(), (100, 255, 0), 2)

                else:
                    is_detected = False
                    time_of_detect_start = None

                cv2.rectangle(frame, mt.motion_roi.tl(), mt.motion_roi.br(), (255, 100, 0), 2)

        cv2.imshow("Motion", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break
        elif key == ord('t'):
            cur_tracking_method = (cur_tracking_method + 1) % len(tracking_methods)
            tracker = None

    mt.clean_up()

if __name__ == "__main__":
    test_object_tracker()
