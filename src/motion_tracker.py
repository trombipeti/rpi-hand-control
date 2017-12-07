#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from opencv_rect import CvRect
from timeit import default_timer as timer
from datetime import datetime

def find_largest_contour(bin_image):
    """
    Finds the contour with the largest area on the given image.
    @param bin_image A binary image on which this function finds the largest contour. It won't be modified.
    @return A vector of points of the largest area contour.
    """

    # OpenCV API 3 changed this function...
    if int(cv2.__version__.split(".")[0]) >= 3:
        _, all_contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        all_contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour_index = -1
    max_contour_area = 0
    for i in range(len(all_contours)):
        if len(all_contours[i]) > 0:
            cur_area = cv2.contourArea(all_contours[i])
            if cur_area > max_contour_area:
                max_contour_area = cur_area
                max_contour_index = i

    return all_contours[max_contour_index] if max_contour_index >= 0 else None



class MotionTracker(object):

    def __init__(self, videosource, filter_alpha = None, motion_threshold = None):

        self.accum_motion = None
        self.prev_frame = None
        self.motion_image = None
        self.motion_roi = None

        # Ez azért ilyen furcsán van, hogy a leszármazott osztályokban is lehessen
        # az alapértelmezett értékeket használni
        self.filter_alpha = filter_alpha
        self.motion_threshold = 10 if motion_threshold is None else motion_threshold

        self.last_fps = 0

        try:
            self.__vid_cap = cv2.VideoCapture(videosource)
            self.__is_inited = True
        except cv2.error as e:
            print("!!!OPENCV EXCEPTION!!!")

    def __del__(self):
        self.clean_up()

    def clean_up(self):
        if self.__vid_cap is not None:
            self.__vid_cap.release()

    def all_input_frames(self):

        num_frames = 0
        last_sec_start = None
        while True and self.__vid_cap is not None:
            if last_sec_start is None:
                last_sec_start = timer()

            ret, frame = self.read_frame()
            if ret:
                num_frames += 1
                time_now = timer()
                if (time_now - last_sec_start) >= 1.0:
                    self.last_fps = num_frames
                    num_frames = 0
                    last_sec_start = time_now

                yield frame
            else:
                raise StopIteration

    def read_frame(self):
        return self.__vid_cap.read() 

    def process_frame(self, frame):

        bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.__update_motion_images(frame)
        self.__update_motion_roi()

    def __update_motion_images(self, frame):

        if self.accum_motion is None:
            self.accum_motion = np.zeros((frame.shape[0], frame.shape[1], 1), np.float32)

        if self.prev_frame is not None:
            # Threshold the difference image with the set value. Note: first return value - the '_' - is rubbish here
            _, motion_thresh_img = cv2.threshold( cv2.absdiff(frame, self.prev_frame),
                                                  self.motion_threshold,
                                                  255,
                                                  cv2.THRESH_BINARY)
            self.motion_image = cv2.cvtColor(motion_thresh_img, cv2.COLOR_BGR2GRAY)
            # If filter_alpha is None, we try some adaptive stuff
            alpha = self.filter_alpha if self.filter_alpha is not None else 0.3 #self.last_fps / 10.0
            # IIR filtering with the set value
            cv2.accumulateWeighted(self.motion_image, self.accum_motion, alpha)

        self.prev_frame = frame

    def get_unfiltered_motion_roi(self):

        accum_copy = np.uint8(self.accum_motion)
        largest_contour = find_largest_contour(accum_copy)
        if(largest_contour is not None):
            return CvRect(cv2.boundingRect(largest_contour))

        return None

    def get_motion_roi(self):
        return self.motion_roi

    def __update_motion_roi(self):

        if self.motion_image is not None:
            cur_roi = self.get_unfiltered_motion_roi()
            if cur_roi is not None:
                if cv2.countNonZero(self.motion_image) >= cur_roi.area() * 0.005:
                    if self.motion_roi is None:
                        self.motion_roi = CvRect((0, 0, 0, 0))
                    _intersection = self.motion_roi.intersect(cur_roi)

                    # Az eltárolt roi-n belül van-e a jelenleg megtalált
                    roi_in_current = (_intersection == cur_roi)

                    # Van-e közös metszetük, de nincs belül
                    roi_expands_current = (_intersection.area() > cur_roi.area() * 0.005 and not roi_in_current)
                    
                    # Az előző roi területének 1/X részénél nem nagyobb
                    roi_too_small = (cur_roi.h <= self.motion_roi.h * 0.7)

                    # Ha "kicsinyedik" a téglalap, akkor megtartjuk, amennyiben nem túl gyors ez a kicsinyedés
                    # Ha van közös metszet, akkor ha nagyobb az eddigi 1/x részénél a jelenlegi ROI, megtartjuk
                    # Ha nincs közös metszet, akkor csak az olyan ROI-t tartjuk meg, ami nagyobb, mint a jelenlegi
                    if ((roi_in_current and not roi_too_small) or
                        (roi_expands_current and cur_roi.area() >= self.motion_roi.area() * 0.2) or
                        (not roi_in_current  and cur_roi.area() > self.motion_roi.area())):
    
                        self.motion_roi = cur_roi.scale_inside(self.motion_image.shape, 1.1)


def test_motion_tracker():

    mt = MotionTracker(0, 0.3, 30)

    cv2.namedWindow("Motion", cv2.WINDOW_NORMAL)
    for frame in mt.all_input_frames():

        mt.process_frame(frame)
        motion_rect = mt.get_motion_roi()

        disp_img = cv2.cvtColor(mt.accum_motion, cv2.COLOR_GRAY2BGR)

        if motion_rect is not None:
            cv2.rectangle(disp_img, motion_rect.tl(), motion_rect.br(), (255, 100, 0), 2)
            pass

        txt = "T {0}, a {1:.2f}".format(mt.motion_threshold, mt.filter_alpha)
        cv2.putText(disp_img, txt, (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Motion", disp_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('p'):
            if cv2.waitKey(-1) & 0xFF == ord('p'):
                timetxt = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = "motion_tracker_t{0}_a{1:.2f}_{2}.png".format(mt.motion_threshold, mt.filter_alpha, timetxt)
                cv2.imwrite(fn, disp_img * 255)
        elif key == ord('w'):
            mt.motion_threshold += 5 if mt.motion_threshold <= 250 else 0
        elif key == ord('s'):
            mt.motion_threshold -= 5 if mt.motion_threshold >= 5 else 0
        elif key == ord('e'):
            mt.filter_alpha += 0.05 if mt.filter_alpha < 0.96 else 0
        elif key == ord('d'):
            mt.filter_alpha -= 0.05 if mt.filter_alpha > 0.06 else 0
        elif key == 32:
            newkey = cv2.waitKey(-1) & 0xFF

    mt.clean_up()

if __name__ == "__main__":
    test_motion_tracker()

