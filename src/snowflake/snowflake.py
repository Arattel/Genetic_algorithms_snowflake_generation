import cv2
import numpy as np
import random

from .generator import TopCutter, CutoutGenerator
from .models import Point, Segment, segments_to_numpy_points
from .utils import *


class Snowflake:
    def __init__(self, genome, random_seed=0):
        random.seed(random_seed)
        self.mh = 1000 // 2
        self.mw = int(2 * self.mh * math.tan(math.pi * 0.08333))
        self.setup(genome, max_cutouts=1000)

    def setup(
            self,
            genome,
            min_cutouts=1,
            max_cutouts=float("inf"),
            max_global_iterations=10,
            max_sub_iterations=500
    ):
        self.reset()
        self.genome = genome
        self.min_cutouts_count = min_cutouts
        self.max_cutouts_count = max_cutouts
        self.maximum_global_iterations = max_global_iterations
        self.maximum_iterations = max_sub_iterations
        self.finished = False

        x = 0
        y = 0

        self.basis = np.array([[(x, y), (x + self.mw, y), (self.mw // 2, y + self.mh)]], dtype=np.int32)

        self.left = Segment(Point(self.mw / 2, y + self.mh), Point(x, y))
        self.right = Segment(Point(x + self.mw, y), Point(self.mw / 2, y + self.mh))
        self.top_cutter = TopCutter(self.left, self.right, genome[4])
        self.cutout_gen = CutoutGenerator(self.genome, self.mw, max(self.mw, self.mh))

    def reset(self):
        self.segments = []
        self.cutouts = []
        self.cutout_steps = []
        self.uncut_steps = []

    def generate(self):
        global_iterations = self.maximum_global_iterations
        while (
                (len(self.cutouts) < self.min_cutouts_count
                 or len(self.cutouts) > self.max_cutouts_count)
                and global_iterations > 0
        ):
            self.reset()

            self.segments.append(Segment(self.left.start, self.left.end))
            self.segments.append(Segment(self.right.start, self.right.end))
            top_segments = self.top_cutter.generate()

            top_points = [ts.start for ts in top_segments]
            top_points.append(top_segments[-1].end)
            top_points.append(self.right.start)

            self.segments[-1].start = top_segments[-1].end

            tmp = [self.segments.pop(0)]
            self.segments = tmp + top_segments + self.segments

            iterations = self.maximum_iterations
            while iterations > 0 and len(self.segments) > 0:
                cutout = self.cutout_gen.generate(self.segments, self.cutouts)
                if cutout is not None:
                    iterations = self.maximum_iterations
                iterations -= 1

            self.top_segments = [Segment(self.left.end, self.right.start), *top_segments]

            global_iterations -= 1

    def draw(self, max_height):
        scale = max_height / 1000
        self.canvas = np.zeros((int(self.mh * scale), int(self.mw * scale)))

        # draw cuts
        cv2.fillPoly(self.canvas, (self.basis * scale).astype(int), (255, 255, 255))
        cutouts = [(c.to_numpy() * scale).astype(int) for c in self.cutouts]
        cutouts.append((segments_to_numpy_points(self.top_segments) * scale).astype(int))
        cv2.fillPoly(self.canvas, cutouts, (0, 0, 0))

        # remove noise
        k = max(3, int(3 * scale))
        self.canvas = cv2.erode(self.canvas, np.ones((1, k), 'uint8'), iterations=1)
        self.canvas = cv2.dilate(self.canvas, np.ones((1, k), 'uint8'), iterations=1)

        # expand canvas
        h, w = self.canvas.shape
        side = 2 * h
        snowflake_img = np.zeros((side, side))
        offset = side // 2 - w // 2
        snowflake_img[:h, offset:offset + w] = self.canvas

        # create rotated parts
        final_img = np.zeros_like(snowflake_img)
        for i in range(6):
            final_img += rotate_image(snowflake_img, i * 60)

        mirror_part = cv2.flip(final_img, 0)
        mirror_part = rotate_image(mirror_part, 30)
        final_img += mirror_part

        # remove noise
        final_img = final_img.astype(np.uint8)
        final_img = cv2.dilate(final_img, np.ones((k, k), 'uint8'), iterations=1)
        final_img = cv2.erode(final_img, np.ones((k, k), 'uint8'), iterations=1)

        return final_img


if __name__ == "__main__":
    # generate snowflake
    s = Snowflake(genome=np.random.rand(5) * 7)
    s.generate()
    final_img = s.draw(max_height=1000)
    cv2.imwrite("tmp.png", final_img)
    cv2.imwrite("tmp_part.png", s.canvas)
