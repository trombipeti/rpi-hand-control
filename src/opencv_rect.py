#!/usr/other.in/env python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import copy

class CvRect:
    """ Osztály OpenCV téglalapok kezelésének megkönnyítésére. """

    def __init__(self, params_tuple = (0,0,0,0)):
        """
        Konstruktor, amely 4 elemű tuple-ből inicializálja az objektumot.
        """

        self.x = params_tuple[0]
        self.y = params_tuple[1]
        self.w = params_tuple[2]
        self.h = params_tuple[3]

    def __str__(self):
        return "{0},{1} {2},{3}".format(self.x, self.y, self.w, self.h)

    def tl(self):
        return (int(self.x), int(self.y))

    def br(self):
        return (int(self.x + self.w), int(self.y + self.h))

    def center(self):
        return (int(self.x + (self.w / 2)),
                int(self.y + (self.h / 2)))

    def area(self):
        return self.w * self.h

    def shifted(self, x, y):
        return CvRect((self.x + x, self.y + y, self.w, self.h))

    def intersect(self, other):
        
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        w = min(self.x + self.w, other.x + other.w) - x
        h = min(self.y + self.h, other.y + other.h) - y
        if w < 0 or h < 0:
            return CvRect((0, 0, 0, 0))
        return CvRect((x, y, w, h))

    def __lt__(self, other):
        """ Egyéni összehasonlító függvény, amely azt adja meg, hogy az egyik téglalap területe kisebb-e a másikénál """
        return self.area() < other.area()

    def __gt__(self, other):
        """ Egyéni összehasonlító függvény, amely azt adja meg, hogy az egyik téglalap területe kisebb-e a másikénál """
        return self.area() > other.area()

    def __eq__(self, other):
        """ Egyéni összehasonlító függvény, amely azt adja meg, hogy az egyik téglalap területe egyenlő-e a másikéval """
        return self.area() == other.area()

    def __iter__(self):
        for i in self.x, self.y, self.w, self.h:
            yield i

    def scaled(self, s):

        if s == 1:
            return copy.deepcopy(self)

        if s == 0:
            return CvRect((0,0,0,0))

        center = (self.x + self.w * 0.5, self.y + self.h * 0.5)
        new_w = self.w * s
        new_h = self.h * s

        r = CvRect()
        r.x = int(center[0] - new_w * 0.5)
        r.y = int(center[1] - new_h * 0.5)
        r.w = int(new_w)
        r.h = int(new_h)
        return r

    def scale_inside(self, rect, scale):

        if scale == 1:
            return self.scaled(1) # we need to return a deep copy!

        scaled = self.scaled(scale)
        # numpy shape-ben forditva van az x meg az y!
        scaled.x = int(max( min(scaled.x, rect[1]), 0))
        scaled.y = int(max( min(scaled.y, rect[0]), 0))
        scaled.w = int(min(scaled.w, rect[1] - scaled.x))
        scaled.h = int(min(scaled.h, rect[0] - scaled.y))

        return scaled
