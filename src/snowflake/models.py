import math
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, p):
        return isinstance(p, Point) and self.x == p.x and self.y == p.y

    def to_numpy(self):
        return np.array([self.x, self.y], dtype=np.int32)


class Segment:
    def __init__(self, start_point, end_point):
        self.start = start_point
        self.end = end_point
    
    def x_len(self):
        return self.end.x - self.start.x
    
    def y_len(self):
        return self.end.y - self.start.y
    
    def len(self):
        return math.sqrt(self.x_len() ** 2 + self.y_len() ** 2)
    
    def ang(self):
        return math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)

    def ang_deg(self):
        return self.ang() * (180 / math.pi)
    
    def rev_ang(self):
        return math.atan2(self.start.y - self.end.y, self.start.x - self.end.x)
    
    def rev_ang_deg(self):
        return self.rev_ang() * (180 / math.pi)
    
    def distance2point(self, p):
        A = p.x - self.start.x
        B = p.y - self.start.y
        C = self.x_len()
        D = self.y_len()

        len_sq = C**2 + D**2
        param = -1
        if len_sq != 0:
            param = A * C + B * D / len_sq
        
        if param < 0:
            xx = self.start.x
            yy = self.start.y
        elif param > 1:
            xx = self.end.x
            yy = self.end.y
        else:
            xx = self.start.x + param * C
            yy = self.start.y + param * D
        
        dx = p.x - xx
        dy = p.y - yy
        return math.sqrt(dx**2 + dy**2)
    
    def to_numpy(self):
        return np.array([
            self.start.to_numpy(),
            self.end.to_numpy()
        ], dtype=np.int32)


class Cutout:
    def __init__(self, side_segment, first_segment, second_segment):
        self.side_segment = side_segment
        self.first_segment = first_segment
        self.second_segment = second_segment

    def points(self):
        return [
            self.side_segment.start,
            self.first_segment.end,
            self.side_segment.end,
            self.side_segment.start
        ]

    def to_numpy(self):
        segments = [
            self.side_segment, 
            self.first_segment, 
            self.second_segment
        ]
        return segments_to_numpy_points(segments)


def segments_to_numpy_points(segments):
        segments = np.array([seg.to_numpy() for seg in segments])
        c, s, p = segments.shape
        points = segments.reshape((c * s, p))
        points = unique_rows(points)
        return np.array(points)


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
