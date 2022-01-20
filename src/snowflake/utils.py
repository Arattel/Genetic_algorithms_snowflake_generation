import cv2
import numpy as np
import math
import random
from .models import Point, Segment


def rand(min_, max_):
    return random.random() * (max_ - min_) + min_


def prob(p):
    return random.random() < p


def dist(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_point_from_start(segment, fraction):
    x_trim = fraction * segment.x_len()
    y_trim = fraction * segment.y_len()
    return Point(
        segment.start.x + x_trim, 
        segment.start.y + y_trim
    )


def get_sub_segment(segment, margin):
    l = segment.len()
    start_edge = margin / l
    end_edge = 1 - start_edge
    return Segment(
        get_point_from_start(segment, start_edge),
        get_point_from_start(segment, end_edge)
    )


def intersect(f_segment, s_segment):
    denom = (
        (s_segment.end.y - s_segment.start.y) * (f_segment.end.x - f_segment.start.x)
    ) - (
        (s_segment.end.x - s_segment.start.x) * (f_segment.end.y - f_segment.start.y)
    )
    nume_a = (
        (s_segment.end.x - s_segment.start.x) * (f_segment.start.y - s_segment.start.y)
    ) - (
        (s_segment.end.y - s_segment.start.y) * (f_segment.start.x - s_segment.start.x)
    )
    nume_b = (
        (f_segment.end.x - f_segment.start.x) * (f_segment.start.y - s_segment.start.y)
    ) - (
        (f_segment.end.y - f_segment.start.y) * (f_segment.start.x - s_segment.start.x)
    )

    if denom == 0:
        return None
    
    u_a = nume_a / denom
    u_b = nume_b / denom

    if u_a >= 0 and u_a <= 1 and u_b >= 0 and u_b <= 1:
        return Point(
            f_segment.start.x + (u_a * (f_segment.end.x - f_segment.start.x)),
            f_segment.start.y + (u_a * (f_segment.end.y - f_segment.start.y))
        )

    return None


def calc_square(p1, p2, p3):
    a = dist(p1, p2)
    b = dist(p2, p3)
    c = dist(p3, p1)
    p = (a + b + c) / 2
    return math.sqrt(p * (p - 1) * (p - b) * (p - c))


def random_point_between(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    ratio = random.random()
    return Point(p1.x + dx * ratio, p1.y + dy * ratio)


def random_point_with_margin(segment, margin):
    ss = get_sub_segment(segment, margin)
    return random_point_between(ss.start, ss.end)


def from_ang_and_len(start_point, ang, len):
    end_point = Point(
        start_point.x + len * math.cos(ang),
        start_point.y + len * math.sin(ang)
    )
    return Segment(start_point, end_point)


def to_rad():
    return math.pi / 180


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
