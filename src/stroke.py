#!/usr/bin/env python
# -*- encoding: utf-8 -*-

## Python port of easystroke's stroke class and related stuff

import math

stroke_infinity = 0.2
EPS = 0.000001

class StrokePoint:
    
    def __init__(self):
        self.x = 0
        self.y = 0
        self.t = 0
        self.dt = 0
        self.alpha = 0 # Hány pi radián

    def __str__(self):
        return "{0},{1}, t={2}, dt={3}, alpha={4}, {5}rad, {6}°".format(self.x, self.y, self.t, self.dt, self.alpha, self.alpha * math.pi, self.alpha * 180.0)

class Stroke:

    def __init__(self, capacity = float("inf")):

        self.points = []
        self.capacity = capacity

    def add_point(self, x, y):
        """ TODO """
        assert self.capacity > len(self.points)

        p = StrokePoint()
        p.x = x
        p.y = y
        self.points.append(p)

    def finish(self):
        assert len(self.points) > 0

        # Időparaméter normalizálása
        total = 0.0
        self.points[0].t = 0.0
        for i in range(0, len(self.points) - 1):
            total += math.hypot(self.points[i + 1].x - self.points[i].x, self.points[i + 1].y - self.points[i].y)
            self.points[i + 1].t = total
        if total == 0.0:
            total = 1.0
        for i in range(1, len(self.points)):
            self.points[i].t /= total

        # Pozíciók normalizálása
        minX = min(self.points, key = lambda p: p.x).x
        minY = min(self.points, key = lambda p: p.y).y
        maxX = max(self.points, key = lambda p: p.x).x
        maxY = max(self.points, key = lambda p: p.y).y

        scale = max(maxX - minX, maxY - minY)
        if scale < 0.001:
            scale = 1
        for p in self.points:
            p.x = (p.x-(minX+maxX)/2)/scale + 0.5
            p.y = (p.y-(minY+maxY)/2)/scale + 0.5

        # Gyorsulás és szög paraméterek kiszámítása
        for i in range(0, len(self.points) - 1):
            self.points[i].dt    = self.points[i + 1].t - self.points[i].t
            self.points[i].alpha = math.atan2( self.points[i + 1].y - self.points[i].y, self.points[i + 1].x - self.points[i].x) / math.pi

    @staticmethod
    def __angle_difference(alpha, beta):
        """ Megadja két szög különbségét radián/pi mértékegységben, -1 és 1 közt """

        d = alpha - beta
        if d <= -1.0:
            d += 2.0
        elif d > 1.0:
            d -= 2.0
        return d

    @staticmethod
    def __step(stroke, other, dist, k, prev_x, prev_y, x, y, x2, y2):
        tx = stroke.points[x].t
        ty = other.points[y].t
        dtx = stroke.points[x2].t - tx;
        dty = other.points[y2].t - ty;
        if dtx >= dty * 2.2 or dty >= dtx * 2.2 or dtx < EPS or dty < EPS:
            return k
        k += 1

        d = 0.0
        i = x
        j = y
        next_tx = (stroke.points[i+1].t - tx) / dtx
        next_ty = (other.points[j+1].t - ty) / dty
        cur_t = 0.0

        while(True):
            ad = Stroke.__angle_difference(stroke.points[i].alpha, other.points[j].alpha)**2
            next_t = min(next_tx, next_ty)
            done = next_t >= 1.0 - EPS
            if done:
                next_t = 1.0
            d += (next_t - cur_t) * ad
            if done:
                break
            cur_t = next_t
            if next_tx < next_ty:
                i += 1
                next_tx = (stroke.points[i + 1].t - tx) / dtx
            else:
                j += 1
                next_ty = (other.points[j + 1].t - ty) / dty

        new_dist = dist[x][y] + d * (dtx + dty)
        if new_dist >= dist[x2][y2]:
            return k
        
        prev_x[x2][y2] = x
        prev_y[x2][y2] = y
        dist[x2][y2] = new_dist

        return k

    def compare(self, other):

        dist = [[stroke_infinity]* len(other.points) for _ in range(len(self.points))]
        dist[0][0] = 0.0

        prev_x = [[0]* len(other.points) for _ in range(len(self.points))]
        prev_y = [[0]* len(other.points) for _ in range(len(self.points))]

        for x in range(0, len(self.points) - 1):
            for y in range(0, len(other.points) - 1):
                if dist[x][y] >= stroke_infinity:
                    continue

                tx = self.points[x].t
                ty = other.points[y].t
                max_x = x
                max_y = y
                k = 0

                while k < 4:
                    if self.points[max_x + 1].t - tx > other.points[max_y + 1].t - ty:
                        max_y += 1
                        if max_y == len(other.points) - 1:
                            k = Stroke.__step(self, other, dist, k, prev_x, prev_y, x, y, len(self.points) - 1, len(other.points) - 1)
                            break
                        for x2 in range(x + 1, max_x + 1):
                            k = Stroke.__step(self, other, dist, k, prev_x, prev_y, x, y, x2, max_y)
                            pass
                    else:
                        max_x += 1
                        if max_x == len(self.points) - 1:
                            k = Stroke.__step(self, other, dist, k, prev_x, prev_y, x, y, len(self.points) - 1, len(other.points) - 1)
                            break
                        for y2 in range(y + 1, max_y + 1):
                            k = Stroke.__step(self, other, dist, k, prev_x, prev_y, x, y, max_x, y2)
                            pass

        return dist[-1][-1] # Jobb alsó sarok

def __test_stroke():
    s = Stroke()
    s.add_point(0, 0)
    s.add_point(1, 1)
    s.add_point(2, 2)
    s.add_point(3, 3)
    s.add_point(4, 4)
    s.finish()

    o = Stroke()
    o.add_point(0, 0)
    o.add_point(4, 0)
    o.finish()

    c = s.compare(o)
    print("Compare: {0}".format(c))
    for p in s.points:
        print(p)

if __name__ == "__main__":
    __test_stroke()