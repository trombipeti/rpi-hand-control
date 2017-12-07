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

class ObjectTracker:

    (cv2_major, cv2_minor, cv2_build) = (int(v) for v in cv2.__version__.split('.'))

    STATUS_HAND_SEARCH = 0
    STATUS_FIST_SEARCH = 1
    STATUS_TRACKING = 2
    STATUS_LOST_TRACKING = 3
    STATUS_MAX = 4

    StatusNames = {0: "HAND", 1: "FIST", 2: "TRACK", 3: "LOST"}

    def __init__(self, palm_cascade, fist_cascade):
        self.palm_classifier = cv2.CascadeClassifier(palm_cascade)
        self.fist_classifier = cv2.CascadeClassifier(fist_cascade)

        self.status = self.STATUS_HAND_SEARCH
        self.tracker = None
        self.time_of_detect_start = None
        self.time_of_tracker_lost = None
        self.hand_position = None

    def get_min_detect_time(self):
        if self.status == self.STATUS_HAND_SEARCH:
            return 0.5
        else:
            return 0.25

    def update_detect_timer(self, aborted = False):
        if aborted:
            self.time_of_detect_start = None
            return False
        elif self.time_of_detect_start is None:
            self.time_of_detect_start = timer()
            return False
        else:
            return timer() - self.time_of_detect_start > self.get_min_detect_time()

    def process_frame(self, full_frame, roi = None):
        if roi is None:
            roi = CvRect((0, 0, full_frame.shape[0], full_frame.shape[1]))
        if self.status < self.STATUS_TRACKING:
            self.hand_position = self.detect_largest_object(image_roi(full_frame, roi), self.status == self.STATUS_FIST_SEARCH)

            if self.update_detect_timer(self.hand_position is None):
                self.status += 1

        elif self.status == self.STATUS_TRACKING:
            self.time_of_detect_start = None
            if self.tracker is None:
                if ObjectTracker.cv2_major >= 3 and ObjectTracker.cv2_minor >= 2:
                    self.tracker = cv2.TrackerKCF_create()
                else:
                    self.tracker = cv2.Tracker_create("KCF")
                hp = self.hand_position
                if roi is not None:
                    hp = hp.shifted(roi.x, roi.y)
                self.tracker.init(full_frame, tuple(hp))
            else:
                ok, r = self.tracker.update(full_frame)
                rectbbox = CvRect(r)
                in_frame_area = rectbbox.intersect(CvRect( (0, 0, full_frame.shape[1], full_frame.shape[0])) ).area()
                if ok and in_frame_area > 0:
                    self.hand_position = rectbbox
                else:
                    self.status = self.STATUS_LOST_TRACKING
                    self.time_of_tracker_lost = timer()
                    self.tracker = None

        elif self.status == self.STATUS_LOST_TRACKING:
            if timer() - self.time_of_tracker_lost > self.get_min_detect_time():
                self.status = self.STATUS_HAND_SEARCH
            handpos = self.detect_largest_object(full_frame)
            fistpos = self.detect_largest_object(full_frame, True)
            if handpos is not None:
                self.status = self.STATUS_HAND_SEARCH
            elif fistpos is not None:
                self.hand_position = fistpos
                self.status = self.STATUS_TRACKING
            pass

    def detect_largest_object(self, img, fist = False):
        objs = self.detect_all_objects(img, fist)
        if objs is not None and len(objs) > 0:
            return sorted(objs, reverse = True)[0]
        else:
            return None

    def detect_all_objects(self, img, fist = False):

        if img is not None and not self.fist_classifier.empty() and not self.palm_classifier.empty():

            try:
                if fist:
                    objects = self.fist_classifier.detectMultiScale(img)
                else:
                    objects = self.palm_classifier.detectMultiScale(img)
            except:
                objects = []

            if len(objects) > 0:
                obj_rects = list()
                for o in objects:
                    obj_rects.append(CvRect(o))

                return obj_rects
            return None

    trackingMethods = ["KCF", "MIL", "TLD"]

    @staticmethod
    def createCVTracker(method):
        if method not in ObjectTracker.trackingMethods:
            raise ValueError("Method must be one of".join(m for m in ObjectTracker.trackingMethods))
        print(method)
        if ObjectTracker.cv2_major >= 3 and ObjectTracker.cv2_minor >= 2:
            return {"KCF": cv2.TrackerKCF_create,
                    "MIL": cv2.TrackerMIL_create,
                    "TLD": cv2.TrackerTLD_create}[method]()
        else:
            return cv2.Tracker_create(method)


def draw_stroke(img, stroke, scale = 1):
    if stroke is not None and len(stroke.points) > 1:
        for i in range(1, len(stroke.points)):
            prevx = stroke.points[i-1].x * scale
            curx = stroke.points[i].x * scale
            prevy = stroke.points[i-1].y * scale
            cury = stroke.points[i].y * scale
            a = (prevx - curx, prevy - cury)
            a = math.sqrt(a[0] * a[0] + a[1] * a[1]) / math.sqrt(scale)
            cv2.line(img, (int(prevx), int(prevy)),
                         (int(curx), int(cury)), (stroke.points[i].t, 1.0 - stroke.points[i].t, 0.0), 3)

def test_object_tracker():

    cur_tracking_method = 0
    stroke_recording_mode = False
    flipped_mode = False
    ref_strokes =  []


#    ot = ObjectTracker(0, "../data/jordan-hands-LBP-24x32-201711071511", 0.3)
    mt = MotionTracker(0, 0.3, 30)
    ot = ObjectTracker("../data/jordan-hands-LBP-32x48-20171107.xml", "../data/fist.xml")
#    ot = ObjectTracker(0, "../data/fist.xml", 0.3)

    cv2.namedWindow("Gesture control", cv2.WINDOW_NORMAL)

    time_of_detect_start = None
    is_tracking = False
    tracker = None
    obj_rect = None
    cur_obj_rect_size = CvRect()
    last_fps = 0
    framecounter = 0
    stroke = None
    for frame in mt.all_input_frames():

        frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
        if flipped_mode:
            frame = cv2.flip(frame, 1)

        framecounter += 1
        if last_fps != mt.last_fps:
                print("{0}; {1}; {2}".format(framecounter, mt.last_fps, int(cur_obj_rect_size.w * cur_obj_rect_size.h)))
                last_fps = mt.last_fps

        if ot.status == ObjectTracker.STATUS_TRACKING:
            ot.process_frame(frame.copy())
            if ot.status == ObjectTracker.STATUS_TRACKING:
                if stroke is not None and ot.hand_position is not None:
                    c = ot.hand_position.center()
                    stroke.add_point(c[0], c[1])

        else:
            if ot.status != ObjectTracker.STATUS_LOST_TRACKING:
                mt.process_frame(frame.copy())
            else:
                mt.motion_roi = None
            ot.process_frame(frame.copy(), mt.motion_roi)
            if ot.status == ObjectTracker.STATUS_TRACKING:
                stroke = Stroke()
            if stroke is not None and ot.hand_position is not None:
                c = ot.hand_position.center()
                stroke.add_point(c[0], c[1])

        if ot.status == ObjectTracker.STATUS_HAND_SEARCH and stroke is not None and len(stroke.points) > 0:
            finished_stroke = stroke
            finished_stroke.finish()
            stroke = None
            if stroke_recording_mode:
                ref_strokes.append(finished_stroke)
                for i in range(len(ref_strokes)):
                    refimg = np.zeros((400, 400, 3), np.float32)
                    draw_stroke(refimg, ref_strokes[i], 400)
                    wn = "Ref {0}".format(i + 1)
                    cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
                    cv2.imshow(wn, refimg)
            else:
                strokeimg = np.zeros((400, 400, 3), np.float32)
                draw_stroke(strokeimg, finished_stroke, 400)

                cv2.namedWindow("Stroke", cv2.WINDOW_NORMAL)
                cv2.imshow("Stroke", strokeimg)
                scores = []
                for r in ref_strokes:
                    score = finished_stroke.compare(r)
                    scores.append(score)
                    print("Compared: {0}".format(score))
                if len(scores) > 0 and min(scores) < 0.2:
                    print("!!!MATCH!!!")
                    match = ref_strokes[scores.index(min(scores))]
                    matchimg = np.zeros((400, 400, 3), np.float32)
                    draw_stroke(matchimg, match, 400)
                    winname = "Match, score: {0}".format(min(scores))
                    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
                    cv2.imshow(winname, matchimg)




        cv2.putText(frame, ObjectTracker.StatusNames[ot.status], (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        draw_hand_pos = ot.hand_position
        if mt.motion_roi is not None:
            draw_hand_pos = draw_hand_pos if draw_hand_pos is None else draw_hand_pos.shifted(mt.motion_roi.x, mt.motion_roi.y)
            cv2.rectangle(frame, mt.motion_roi.tl(), mt.motion_roi.br(), (255, 100, 0), 2)

        if draw_hand_pos is not None:
            cv2.rectangle(frame, draw_hand_pos.tl(), draw_hand_pos.br(), (0, 100, 255), 2)
        
        if stroke is not None and len(stroke.points) > 1 and ot.status >= ObjectTracker.STATUS_TRACKING:
            for i in range(1, len(stroke.points)):
                a = (stroke.points[i-1].x - stroke.points[i].x, stroke.points[i-1].y - stroke.points[i].y)
                a = math.sqrt(a[0] * a[0] + a[1] * a[1])
                cv2.line(frame, (int(stroke.points[i-1].x), int(stroke.points[i-1].y)),
                             (int(stroke.points[i].x), int(stroke.points[i].y)), (int(a * 5) % 255, 0, int(a * 5) % 255), 3)

        if stroke_recording_mode:
            cv2.putText(frame, "Record", (int(frame.shape[0] / 2), 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Gesture control", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            break
        elif key == ord('r'):
            stroke_recording_mode = not stroke_recording_mode
        elif key == ord('f'):
            flipped_mode = not flipped_mode

        ####################################x

        # if is_tracking:
        #     if tracker is None:
        #         print("{0}; {1}; {2}; REINIT".format(framecounter, mt.last_fps, int(cur_obj_rect_size.w * cur_obj_rect_size.h)))
        #         tracker = ObjectTracker.createCVTracker(ObjectTracker.trackingMethods[cur_tracking_method])
        #         tracker.init(frame, obj_rect)
        #         cur_stroke = Stroke()
        #         center = CvRect(obj_rect).center()
        #         cur_stroke.add_point(center[0], center[1])

        #         ok = True

        #     else:
        #         ok, obj_rect = tracker.update(frame)

        #     rectbbox = CvRect(obj_rect)
        #     cur_timer = timer()

        #     in_frame_area = rectbbox.intersect(CvRect( (0, 0, frame.shape[1], frame.shape[0])) ).area()


        #     if ok and in_frame_area >= rectbbox.area():
        #         if len(cur_stroke.points) > 100:
        #             cur_stroke.points.pop(0)
        #             # cur_stroke.finish()
        #         cur_stroke.add_point(rectbbox.center()[0], rectbbox.center()[1])
        #         for i in range(1, len(cur_stroke.points)):
        #             a = (cur_stroke.points[i-1].x - cur_stroke.points[i].x, cur_stroke.points[i-1].y - cur_stroke.points[i].y)
        #             a = math.sqrt(a[0] * a[0] + a[1] * a[1])
        #             cv2.line(frame, (int(cur_stroke.points[i-1].x), int(cur_stroke.points[i-1].y)),
        #                             (int(cur_stroke.points[i].x), int(cur_stroke.points[i].y)), (int(a * 5) % 255, 0, int(a * 5) % 255), 3)

        #         cv2.rectangle(frame, rectbbox.tl(), rectbbox.br(), (0, 100, 255), 2)

        #         if rectbbox.area() != cur_obj_rect_size.area():
        #             cur_obj_rect_size = rectbbox
        #             print("{0}; {1}; {2}".format(framecounter, mt.last_fps, int(cur_obj_rect_size.w * cur_obj_rect_size.h)))

        #     else:
        #         is_tracking = False
        #         tracker = None


        # else:
        #     mt.process_frame(frame.copy())

        #     if mt.motion_roi is not None and mt.motion_roi.area() > 0:
        #         objs = ot.detect_all_objects(image_roi(mt.prev_frame, mt.motion_roi))
            
        #         if objs is not None:
        #             num_objs = len(objs)
        #             sorted_rects = sorted(objs, reverse = True)

        #             if num_objs >= 1 and time_of_detect_start is None:
        #                 time_of_detect_start = timer()
        #             elif num_objs == 0:
        #                 time_of_detect_start = None

        #             prev_detected = is_tracking
        #             now_time = timer()
        #             if (time_of_detect_start is not None and
        #                 (now_time - time_of_detect_start) > 0.5):
        #                 is_tracking = True
        #             elif time_of_detect_start is None:
        #                 is_tracking = False


        #             if is_tracking and not prev_detected:
        #                 pass
        #                 # Popen(["xdg-open", "../data/success-kid.jpg"])

        #             if num_objs >= 1:
        #                 obj_rect = sorted_rects[0].shifted(mt.motion_roi.x, mt.motion_roi.y)
        #                 cv2.rectangle(frame, obj_rect.tl(), obj_rect.br(), (100, 255, 0), 2)
        #                 obj_rect = tuple(obj_rect)

        #         else:
        #             is_tracking = False
        #             time_of_detect_start = None

        #         cv2.rectangle(frame, mt.motion_roi.tl(), mt.motion_roi.br(), (255, 100, 0), 2)

        # cv2.imshow("Motion", frame)
        # key = cv2.waitKey(1) & 0xFF
        
        # if key == 27:
        #     break
        # elif key == ord('t'):
        #     cur_tracking_method = (cur_tracking_method + 1) % len(ObjectTracker.trackingMethods)
        #     tracker = None

    mt.clean_up()

if __name__ == "__main__":
    test_object_tracker()
