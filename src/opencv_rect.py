#!/usr/other.in/env python

import cv2
import numpy as np
import copy

class CvRect:

    def __init__(self, params_tuple = (0,0,0,0)):

        self.x = params_tuple[0]
        self.y = params_tuple[1]
        self.w = params_tuple[2]
        self.h = params_tuple[3]

    def __str__(self):
        return "{0},{1} {2},{3}".format(self.x, self.y, self.w, self.h)

    def tl(self):
        return (self.x, self.y)

    def br(self):
        return (self.x + self.w, self.y + self.h)

    def area(self):
        return self.w * self.h

    def intersect(self, other):
        
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        w = min(self.x + self.w, other.x + other.w) - x
        h = min(self.y + self.h, other.y + other.h) - y
        if w < 0 or h < 0:
            return CvRect((0, 0, 0, 0))
        return CvRect((x, y, w, h))

    def scaled(self, s):

        if s == 1:
            return copy.deepcopy(self)

        if s == 0:
            return CvRect((0,0,0,0))

        center = (self.x + self.w * 0.5, self.y + self.h * 0.5)
        new_w = self.w * s
        new_h = self.h * s

        r = CvRect()
        r.x = center[0] - new_w * 0.5
        r.y = center[1] - new_h * 0.5
        r.w = new_w
        r.h = new_h
        return r

    def scale_inside(self, rect, scale):
        scaled = self.scaled(scale)

        return scaled