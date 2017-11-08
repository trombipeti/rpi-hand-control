#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from opencv_rect import CvRect
from motion_tracker import MotionTracker
from stroke import *
from timeit import default_timer as timer
from subprocess import Popen
import math

def image_roi(img, roi_CvRect):
    return img[
                roi_CvRect.y:(roi_CvRect.y + roi_CvRect.h),
                roi_CvRect.x:(roi_CvRect.x + roi_CvRect.w)
               ]

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

    tracking_methods = {
                        0 : "KCF", #: "cv2.TrackerKCF_create",
                        1 : "MIL", #: "cv2.TrackerMIL_create",
                        2 : "BOOSTING", #: "cv2.TrackerBOOSTING_create",
                        3 : "TLD", #: "cv2.TrackerTLD_create",
                        4 : "MEDIANFLOW", #: "cv2.TrackerMEDIANFLOW_create"
                        }
    cur_tracking_method = 0


    ot = ObjectTracker(0, "../data/jordan-hands-LBP-32x48-20171107.xml", 0.3)

    cv2.namedWindow("Motion", cv2.WINDOW_NORMAL)

    time_of_detect_start = None
    is_detected = False
    tracker = None
    obj_rect = None
    last_fps = 0
    for frame in ot.all_input_frames():

        if last_fps != ot.last_fps:
                print("FPS: {0}".format(ot.last_fps))
                last_fps = ot.last_fps

        if is_detected:
            if tracker is None:
                print("Initializing tracker, method: {0}, rect: {1}".format(tracking_methods[cur_tracking_method], obj_rect))
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, obj_rect)
                cur_stroke = Stroke()
                center = CvRect(obj_rect).center()
                cur_stroke.add_point(center[0], center[1])

                ok = True

            else:
                ok, obj_rect = tracker.update(frame)

            rectbbox = CvRect(obj_rect)
            cur_timer = timer()
            if len(cur_stroke.points) > 100:
                cur_stroke.points.pop(0)
                # cur_stroke.finish()
            cur_stroke.add_point(rectbbox.center()[0], rectbbox.center()[1])

            in_frame_area = rectbbox.intersect(CvRect( (0, 0, frame.shape[1], frame.shape[0])) ).area()


            if ok and in_frame_area >= rectbbox.area():
                for i in range(1, len(cur_stroke.points)):
                    a = (cur_stroke.points[i-1].x - cur_stroke.points[i].x, cur_stroke.points[i-1].y - cur_stroke.points[i].y)
                    a = math.sqrt(a[0] * a[0] + a[1] * a[1])
                    cv2.line(frame, (int(cur_stroke.points[i-1].x), int(cur_stroke.points[i-1].y)),
                                    (int(cur_stroke.points[i].x), int(cur_stroke.points[i].y)), (int(a * 5) % 255, 0, int(a * 5) % 255), 3)

                cv2.rectangle(frame, rectbbox.tl(), rectbbox.br(), (0, 100, 255), 2)
            else:
                is_detected = False
                tracker = None


        else:
            ot.process_frame(frame)

            if ot.motion_roi is not None and ot.motion_roi.area() > 0:
                objs = ot.detectAllObjects()
            
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
                        (now_time - time_of_detect_start) > 0.5):
                        is_detected = True
                    elif time_of_detect_start is None:
                        is_detected = False


                    if is_detected and not prev_detected:
                        pass
                        # Popen(["xdg-open", "../data/success-kid.jpg"])

                    if num_objs >= 1:
                        obj_rect = sorted_rects[0].shifted(ot.motion_roi.x, ot.motion_roi.y)
                        cv2.rectangle(frame, obj_rect.tl(), obj_rect.br(), (100, 255, 0), 2)
                        obj_rect = tuple(obj_rect)

                else:
                    is_detected = False
                    time_of_detect_start = None

                cv2.rectangle(frame, ot.motion_roi.tl(), ot.motion_roi.br(), (255, 100, 0), 2)

        cv2.imshow("Motion", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break
        elif key == ord('t'):
            cur_tracking_method = (cur_tracking_method + 1) % len(tracking_methods)
            tracker = None

    ot.clean_up()

if __name__ == "__main__":
    test_object_tracker()
