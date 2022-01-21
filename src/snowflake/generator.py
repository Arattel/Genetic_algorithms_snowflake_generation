from .models import Segment, Cutout
from .utils import *


MIN_ANG = 15
MAX_ANG = 45
FULL_CUT_PROB = 0.1
FULL_CUT_MIN_FRAC = 0.25
FULL_CUT_MAX_FRAC = 0.5
PART_CUT_MIN_LEN = 0.3
PART_CUT_MAX_LEN = 0.8
MAX_SLANT_ANG = 60


class TopCutter:
    def __init__(self, left_segment, right_segment, seed):
        self.left = left_segment
        self.right = right_segment
        self.top = Segment(left_segment.end, right_segment.start)
        self.rnd = random.Random(seed)

    def generate(self):
        if prob(FULL_CUT_PROB):
            top_last_pt = get_point_from_start(
                self.right, 
                rand(FULL_CUT_MIN_FRAC, FULL_CUT_MAX_FRAC, self.rnd)
            )
            return [Segment(self.left.end, top_last_pt)]

        top_first_ang = self.left.rev_ang_deg() - rand(MIN_ANG, MAX_ANG, self.rnd)
        top_first_len = self.top.len() * rand(PART_CUT_MIN_LEN, PART_CUT_MAX_LEN, self.rnd)
        top_first = from_ang_and_len(self.left.end, top_first_ang * to_rad(), top_first_len)
        top_to_intersect = from_ang_and_len(self.left.end, top_first_ang * to_rad(), self.left.len())
        int_pt = intersect(self.right, top_to_intersect)

        next = Segment(top_first.end, random_point_between(self.right.start, int_pt, self.rnd))
        return [top_first, next]


class CutoutGenerator:
    def __init__(self, genome, w, longest):
        self.margin = (w * 0.08) / genome[0] ** 0.5
        self.inner_margin = (self.margin * 2) / genome[0] ** 0.25
        self.point_to_segment_margin = self.margin / genome[0] ** 0.25

        self.min_edge_segment_len = (w * 0.1) / genome[1]
        self.max_edge_semgent_len = (w * 0.5) / max(1, genome[1] ** 0.25)

        self.min_cutout_sq = (w**2 * 0.04) / genome[2]
        self.max_cutout_sq = (w**2 * 0.1) / max(1, genome[2] ** 0.25)

        self.max_projection = longest * 2
        self.min_cutout_lenght = self.min_edge_segment_len / 2
        self.max_cutout_stretch = 5 * max(1, genome[3])
    
    def generate(self, segments, cutouts):
        segment_index = self.select_segment(segments)
        if segment_index == -1:
            return None
        segment = segments[segment_index]

        center_point = random_point_with_margin(segment, self.margin + self.min_edge_segment_len / 2)
        shortest_side = min(
            dist(center_point, segment.start), 
            dist(center_point, segment.end)
        )
        half_edge_segment_len = rand(
            self.min_edge_segment_len / 2,
            min(shortest_side - self.margin, self.max_edge_semgent_len / 2)
        )

        segment_ang = segment.ang_deg()
        normal = segment_ang + 90
        slant_ang = rand(normal - MAX_SLANT_ANG, normal + MAX_SLANT_ANG)

        top_half = from_ang_and_len(center_point, segment_ang * to_rad(), half_edge_segment_len)
        bottom_half = from_ang_and_len(center_point, (segment_ang + 180) * to_rad(), half_edge_segment_len)
        edge_segment = Segment(bottom_half.end, top_half.end)
        full_edge_seg_len = edge_segment.len()

        all_seg = [s for s in segments if s != segment]
        for c in cutouts:
            all_seg.extend((c.first_segment, c.second_segment))
        
        steps = math.ceil((full_edge_seg_len * 2) / self.min_edge_segment_len)
        if steps == 0.0 or steps == 0:
            steps = 1
        step_size = full_edge_seg_len / steps
        max_len = float("inf")

        for i in range(steps):
            start_proj = get_point_from_start(edge_segment, (i * step_size) / full_edge_seg_len)
            proj = from_ang_and_len(start_proj, slant_ang * to_rad(), self.max_projection)
            min_dist = self.shortest_dist(all_seg, proj)
            if min_dist < max_len:
                max_len = min_dist
        
        if max_len > self.max_projection:
            return None
        max_len -= self.inner_margin

        max_len = min(
            max_len,
            self.max_cutout_sq / half_edge_segment_len,
            self.max_cutout_stretch * full_edge_seg_len
        )
        min_len = max(self.min_cutout_lenght, self.min_cutout_sq / half_edge_segment_len)

        if max_len < min_len:
            return None
        
        point_distance_check = False
        cutout_length = rand(min_len, max_len)
        cutout_main = from_ang_and_len(center_point, slant_ang * to_rad(), cutout_length)
        top_segment = None
        bottom_segment = None

        while not point_distance_check:
            if cutout_length < min_len:
                return None
            
            main_point = cutout_main.end
            min_dist_ps = self.shortest_dist_between_point_and_segment(all_seg, main_point)
            if min_dist_ps < self.point_to_segment_margin:
                cutout_length -= self.point_to_segment_margin
                cutout_main = from_ang_and_len(center_point, slant_ang * to_rad(), cutout_length)
            else:
                all_points = [point for s in segments for point in (s.start, s.end) if s != segment]
                for c in cutouts:
                    all_points.append(c.first_segment.end)

                top_segment = Segment(edge_segment.end, cutout_main.end)
                bottom_segment = Segment(edge_segment.start, cutout_main.end)
                min_dist_sp = min(
                    self.shortest_dist_between_points_and_segments(top_segment, all_points),
                    self.shortest_dist_between_points_and_segments(bottom_segment, all_points)
                )

                if min_dist_sp < self.point_to_segment_margin:
                    cutout_length -= self.point_to_segment_margin
                    cutout_main = from_ang_and_len(center_point, slant_ang * to_rad(), cutout_length)
                else:
                    point_distance_check = True
        
        top_segment = top_segment or Segment(edge_segment.end, cutout_main.end)
        bottom_segment = bottom_segment or Segment(edge_segment.start, cutout_main.end)
        
        start_segment = Segment(segment.start, edge_segment.start)
        end_segment = Segment(edge_segment.end, segment.end)
        segments.pop(segment_index)
        segments.insert(segment_index, start_segment)
        segments.insert(segment_index + 1, end_segment)
        
        cutout = Cutout(edge_segment, top_segment, bottom_segment)
        cutouts.append(cutout)
        return cutout

    def select_segment(self, segments):
        suitable_segments = [s for s in segments if s.len() >= self.margin * 2 + self.min_edge_segment_len]
        if not suitable_segments:
            return -1
        
        sum_length = sum((s.len() for s in suitable_segments))
        r = rand(0, sum_length)
        current_length = 0
        for segment in suitable_segments:
            current_length += segment.len()
            if r <= current_length:
                return segments.index(segment)
        
        return -1
    
    def shortest_dist(self, segments, proj_segment):
        min_dist = float("inf")
        for segment in segments:
            i = intersect(segment, proj_segment)
            if i is not None:
                d = dist(i, proj_segment.start)
                if d < min_dist:
                    min_dist = d
        
        return min_dist
    
    def shortest_dist_between_point_and_segment(self, segments, point):
        return min((s.distance2point(point) for s in segments))
    
    def shortest_dist_between_points_and_segments(self, segment, points):
        return min((segment.distance2point(p) for p in points))
