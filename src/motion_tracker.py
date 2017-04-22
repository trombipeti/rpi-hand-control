#!/usr/bin/env python

import cv2
import numpy as np
from opencv_rect import CvRect

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
            if cv2.contourArea(all_contours[i]) > max_contour_area:
                max_contour_index = i

    return all_contours[max_contour_index] if max_contour_index >= 0 else None



class MotionTracker:

    def __init__(self, videosource, filter_alpha = None, motion_threshold = None):

        self.accum_motion = None
        self.prev_frame = None
        self.motion_image = None
        self.motion_roi = CvRect((0, 0, 0, 0))

        # Ez azért ilyen furcsán van, hogy a leszármazott osztályokban is lehessen
        # az alapértelmezett értékeket használni
        self.filter_alpha = 0.3 if filter_alpha is None else filter_alpha
        self.motion_threshold = 20 if motion_threshold is None else motion_threshold
        print(self.filter_alpha, self.motion_threshold)

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
        while True and self.__vid_cap is not None:
            ret, frame = self.__vid_cap.read()
            if ret:
                yield frame
            else:
                raise StopIteration

    def read_frame(self):
        return self.__vid_cap.read() 

    def process_frame(self, frame):

        bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.__update_motion_images(bw_frame)
        self.__update_motion_roi()

    def __update_motion_images(self, frame):

        if self.accum_motion is None:
            self.accum_motion = np.zeros(frame.shape, np.float32)

        if self.prev_frame is not None:
            # Threshold the difference image with the set value. Note: first return value - the '_' - is rubbish here
            _, self.motion_image = cv2.threshold( cv2.absdiff(frame, self.prev_frame),
                                                  self.motion_threshold,
                                                  255,
                                                  cv2.THRESH_BINARY)

            # IIR filtering with the set value
            cv2.accumulateWeighted(self.motion_image, self.accum_motion, self.filter_alpha)

        self.prev_frame = frame

    def get_motion_roi(self):

        accum_copy = np.uint8(self.accum_motion)
        largest_contour = find_largest_contour(accum_copy)
        if(largest_contour is not None):
            return CvRect(cv2.boundingRect(largest_contour))

        return None

    def __update_motion_roi(self):

        if self.motion_image is not None:
            cur_roi = self.get_motion_roi()
            if cur_roi is not None:
                if cv2.countNonZero(self.motion_image) >= cur_roi.area() * 0.005:

                    _intersection = self.motion_roi.intersect(cur_roi)

                    # Az eltárolt roi-n belül van-e a jelenleg megtalált
                    roi_in_current = (_intersection == cur_roi)

                    # Van-e közös metszetük, de nincs belül
                    roi_expands_current = (_intersection.area() > 0 and not roi_in_current)
                    
                    # Az előző roi területének 1/X részénél nem nagyobb
                    roi_too_small = (cur_roi.area() <= self.motion_roi.area() * 0.05)

                    # Ha "kicsinyedik" a téglalap, akkor megtartjuk, amennyiben nem túl gyors ez a kicsinyedés
                    # Ha van közös metszet, akkor ha nagyobb az eddigi 1/x részénél a jelenlegi ROI, megtartjuk
                    # Ha nincs közös metszet, akkor csak az olyan ROI-t tartjuk meg, ami nagyobb, mint a jelenlegi
                    if ((roi_in_current and not roi_too_small) or
                        (roi_expands_current and cur_roi.area() >= self.motion_roi.area() * 0.1) or
                        (not roi_in_current  and cur_roi.area() > self.motion_roi.area())):

                        self.motion_roi = cur_roi.scale_inside(self.motion_image.shape, 1.1)


def test_motion_tracker():

    mt = MotionTracker(0)

    cv2.namedWindow("Motion", cv2.WINDOW_NORMAL)
    for frame in mt.all_input_frames():

        mt.process_frame(frame)
        motion_rect = mt.motion_roi

        disp_img = mt.accum_motion.copy()

        if motion_rect is not None:
            cv2.rectangle(disp_img, motion_rect.tl(), motion_rect.br(), (255, 100, 0), 2)
            pass

        cv2.imshow("Motion", disp_img)
        if cv2.waitKey(1) & 0xFF == 27:
           break

    mt.clean_up()

if __name__ == "__main__":
    test_motion_tracker()