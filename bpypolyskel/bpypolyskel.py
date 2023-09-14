"""
https://github.com/prochitecture/bpypolyskel

Implementation of the straight skeleton algorithm as described by Felkel and Obdržálek in their 1998 conference paper
'Straight skeleton implementation'.

The code for skeletonize() has been ported from the implementation by Botffy at https://github.com/Botffy/polyskel,
in order to be able to use it in Blender. The main changes are:

- The order of the vertices of the polygon has been changed to a right-handed coordinate system
  (as used in Blender). The positive x and y axes point right and up, and the z axis points into
  your face. Positive rotation is counterclockwise around the z-axis.
- The geometry objects used from the library euclid3 in the implementation of Bottfy have been
  replaced by objects based on mathutils.Vector. These objects are defined in the new library bpyeuclid.
- The signature of skeletonize() has been changed to lists of edges for the polygon and eventual hole.
  These are of type Edge2, defined in bpyeuclid.
- Some parts of the skeleton computations have been changed to fix errors produced by the original implementation.
- Algorithms to merge clusters of skeleton nodes and to filter ghost edges have been added.
- A pattern matching algorithm to detect apses, that creates a multi-edge event to create a proper apse skeleton.
"""

import heapq
import os
import re
from collections import Counter, namedtuple
from itertools import chain, combinations, cycle, islice, tee

if os.environ.get('CUSTOM_MATHUTILS') == '1':
    # Use the custom mathutils from 'lib' directory.
    from lib import mathutils
else:
    # Use the standard mathutils package (if it exists).
    try:
        import mathutils
    except ImportError:
        from lib import mathutils

from .bpyeuclid import Edge2, Line2, Ray2, fit_circle_3_points, intersect
from .poly2FacesGraph import Poly2FacesGraph

EPSILON = 0.00001

# Set this value to 1-cos(alpha), where alpha is the largest angle
# between lines to accept them as parallelaccepted as 'parallel'.
PARALLEL = 0.01

APSE_PATTERN = re.compile(r"(L){6,}")
DORMER_PATTERN = re.compile(r"(?=(RLLR))")

# Add a key to enable debug output. For example: debug_outputs["skeleton"] = 1.
# Then the Python list <skeleton> will be added to <debug_outputs> with the key <skeleton>
# in the function 'skeletonize(..)'.
debug_outputs = {}


def iter_circular_prev_next(items):
    prevs, nexts = tee(items)
    prevs = islice(cycle(prevs), len(items) - 1, None)
    return zip(prevs, nexts)


def iter_circular_prev_this_next(items):
    prevs, this, nexts = tee(items, 3)
    prevs = islice(cycle(prevs), len(items) - 1, None)
    nexts = islice(cycle(nexts), 1, None)
    return zip(prevs, this, nexts)


def approx_equals(a, b):
    return a == b or ((a - b).magnitude <= max(a.magnitude, b.magnitude) * 0.001)


class SplitEvent(namedtuple('SplitEvent', 'distance, intersection_point, vertex, opposite_edge')):
    __slots__ = ()

    def __lt__(self, other):
        return self.distance < other.distance


class EdgeEvent(namedtuple('EdgeEvent', 'distance intersection_point vertex_a vertex_b')):
    __slots__ = ()

    def __lt__(self, other):
        return self.distance < other.distance


class DormerEvent:
    def __init__(self, distance, intersection_point, event_list):
        self.distance = distance
        self.intersection_point = intersection_point
        self.event_list = event_list

    def __lt__(self, other):
        return self.distance < other.distance

    def __str__(self):
        return "DormerEvent:%4d d=%4.2f, ip=%s" % (self.id, self.distance, self.intersection_point)

    def __repr__(self):
        return "DormerEvent:%4d d=%4.2f, ip=%s" % (self.id, self.distance, self.intersection_point)


OriginalEdge = namedtuple('OriginalEdge', 'edge bisector_prev, bisector_next')
Subtree = namedtuple('Subtree', 'source, height, sinks')


class LAVertex:
    def __init__(self, point, edge_prev, edge_next, direction_vectors=None, force_convex=False):
        # 'point' is the vertex V(i).
        # 'edge_prev' is the edge from vertex V(i-1) to V(i).
        # 'edge_next' is the edge from vertex V(i) to V(i+1).

        self.point = point
        self.edge_prev = edge_prev
        self.edge_next = edge_next
        self.prev = None
        self.next = None
        self.lav = None

        # TODO: this might be handled better. Maybe membership in lav implies validity?
        self._valid = True

        # 'creator_vectors' are unit vectors: (V(i) to V(i-1), V(i) to V(i+1)).
        creator_vectors = (edge_prev.norm * -1, edge_next.norm)

        if direction_vectors is None:
            direction_vectors = creator_vectors

        dv0 = direction_vectors[0]
        dv1 = direction_vectors[1]
        self._is_reflex = dv0.cross(dv1) > 0

        if force_convex:
            self._is_reflex = False

        op_add_result = creator_vectors[0] + creator_vectors[1]
        self._bisector = Ray2(self.point, op_add_result * (-1 if self._is_reflex else 1))

    def invalidate(self):
        if self.lav is not None:
            self.lav.invalidate(self)
        else:
            self._valid = False

    @property
    def bisector(self):
        return self._bisector

    @property
    def is_reflex(self):
        return self._is_reflex

    @property
    def original_edges(self):
        return self.lav._slav._original_edges

    @property
    def is_valid(self):
        return self._valid

    def next_event(self):
        events = []

        if self.is_reflex:
            # A reflex vertex may generate a split event.
            # Split events happen when a vertex hits an opposite edge, splitting the polygon in two.
            for edge in self.original_edges:
                if edge.edge == self.edge_prev or edge.edge == self.edge_next:
                    continue

                # A potential b is at the intersection of between our own bisector and the bisector of the
                # angle between the tested edge and any one of our own edges.

                # We choose the "less parallel" edge (in order to exclude a potentially parallel edge).
                prev_dot = abs(self.edge_prev.norm.dot(edge.edge.norm))
                next_dot = abs(self.edge_next.norm.dot(edge.edge.norm))
                self_edge = self.edge_prev if prev_dot < next_dot else self.edge_next

                i = Line2(self_edge).intersect(Line2(edge.edge))

                if i is not None and not approx_equals(i, self.point):
                    # Locate candidate b.
                    linvec = (self.point - i).normalized()
                    edvec = edge.edge.norm

                    if abs(self.bisector.v.cross(linvec) - 1.0) < EPSILON:
                        linvec = (self.point - i + edvec * 0.01).normalized()

                    if self.bisector.v.cross(linvec) < 0:
                        edvec = -edvec

                    bisecvec = edvec + linvec

                    if not bisecvec.magnitude:
                        continue

                    bisector = Line2(i, bisecvec, 'pv')
                    b = bisector.intersect(self.bisector)

                    if b is None:
                        continue

                    # Check eligibility of 'b'. A valid 'b' should lie within the area
                    # limited by the edge and the bisectors of its two vertices.
                    x_prev = (
                        (edge.bisector_prev.v.normalized()).cross(
                            (b - edge.bisector_prev.p).normalized()
                        )
                    ) < EPSILON
                    x_next = (
                        (edge.bisector_next.v.normalized()).cross(
                            (b - edge.bisector_next.p).normalized()
                        )
                    ) > -EPSILON
                    x_edge = (edge.edge.norm.cross((b - edge.edge.p1).normalized())) > -EPSILON

                    if not (x_prev and x_next and x_edge):
                        # Candidate discarded
                        continue

                    # Found valid candidate.
                    events.append(SplitEvent(Line2(edge.edge).distance(b), b, self, edge.edge))

        i_prev = self.bisector.intersect(self.prev.bisector)
        i_next = self.bisector.intersect(self.next.bisector)

        if i_prev is not None:
            events.append(
                EdgeEvent(Line2(self.edge_prev).distance(i_prev), i_prev, self.prev, self)
            )
        if i_next is not None:
            events.append(
                EdgeEvent(Line2(self.edge_next).distance(i_next), i_next, self, self.next)
            )

        if not events:
            return None

        ev = min(events, key=lambda event: (self.point - event.intersection_point).magnitude)

        # Generated new event.
        return ev


class LAV:
    def __init__(self, slav):
        self.head = None
        self._slav = slav
        self._len = 0

    @classmethod
    # 'edge_contour' is a list of edges of class Edge2.
    def from_polygon(cls, edge_contour, slav):
        lav = cls(slav)

        for prev, next in iter_circular_prev_next(edge_contour):
            # V(i) is the current vertex.
            # 'prev' is the edge from vertex V(i-1) to V(i).
            # 'this' is the edge from vertex V(i-) to V(i+1).
            lav._len += 1
            vertex = LAVertex(next.p1, prev, next)
            vertex.lav = lav

            if lav.head is None:
                lav.head = vertex
                vertex.prev = vertex.next = vertex
            else:
                vertex.next = lav.head
                vertex.prev = lav.head.prev
                vertex.prev.next = vertex
                lav.head.prev = vertex

        return lav

    @classmethod
    def from_chain(cls, head, slav):
        lav = cls(slav)
        lav.head = head

        for vertex in lav:
            lav._len += 1
            vertex.lav = lav

        return lav

    def invalidate(self, vertex):
        assert vertex.lav is self, "Tried to invalidate a vertex that's not mine"

        vertex._valid = False

        if self.head == vertex:
            self.head = self.head.next

        vertex.lav = None

    def unify(self, vertex_a, vertex_b, point):
        replacement = LAVertex(
            point,
            vertex_a.edge_prev,
            vertex_b.edge_next,
            (vertex_b.bisector.v.normalized(), vertex_a.bisector.v.normalized()),
        )
        replacement.lav = self

        if self.head in [vertex_a, vertex_b]:
            self.head = replacement

        vertex_a.prev.next = replacement
        vertex_b.next.prev = replacement
        replacement.prev = vertex_a.prev
        replacement.next = vertex_b.next

        vertex_a.invalidate()
        vertex_b.invalidate()

        self._len -= 1
        return replacement

    def __len__(self):
        return self._len

    def __iter__(self):
        curr = self.head

        while True:
            yield curr
            curr = curr.next

            if curr == self.head:
                return


class SLAV:
    def __init__(self, edge_contours):
        self._lavs = [LAV.from_polygon(edge_contour, self) for edge_contour in edge_contours]

        # Store original polygon edges for calculation of split events.
        self._original_edges = [
            OriginalEdge(vertex.edge_prev, vertex.prev.bisector, vertex.bisector)
            for vertex in chain.from_iterable(self._lavs)
        ]

    def __iter__(self):
        yield from self._lavs

    def empty(self):
        return not self._lavs

    def handle_dormer_event(self, event):
        # Handle split events (indices 0 and 1).
        ev_prev = event.event_list[0]
        ev_next = event.event_list[1]
        ev_edge = event.event_list[2]
        v_prev = ev_prev.vertex
        v_next = ev_next.vertex

        lav = ev_prev.vertex.lav
        if lav is None:
            return ([], [])

        to_remove = [v_prev, v_prev.next, v_next, v_next.prev]
        lav.head = v_prev.prev

        v_prev.prev.next = v_next.next
        v_next.next.prev = v_prev.prev

        new_lav = [LAV.from_chain(lav.head, self)]
        self._lavs.remove(lav)
        self._lavs.append(new_lav[0])

        p = v_prev.bisector.intersect(v_next.bisector)
        arcs = []

        # From edge event.
        arcs.append(
            Subtree(
                ev_edge.intersection_point,
                ev_edge.distance,
                [ev_edge.vertex_a.point, ev_edge.vertex_b.point, p],
            )
        )

        # From split events.
        arcs.append(
            Subtree(p, (ev_prev.distance + ev_next.distance) / 2.0, [v_prev.point, v_next.point])
        )

        for v in to_remove:
            v.invalidate()

        return (arcs, [])

    def handle_edge_event(self, event):
        sinks = []
        events = []
        lav = event.vertex_a.lav

        if event.vertex_a.prev == event.vertex_b.next:
            # Peak event at intersection.
            self._lavs.remove(lav)

            for vertex in list(lav):
                sinks.append(vertex.point)
                vertex.invalidate()
        else:
            # Edge event at intersection.
            new_vertex = lav.unify(event.vertex_a, event.vertex_b, event.intersection_point)

            if lav.head in (event.vertex_a, event.vertex_b):
                lav.head = new_vertex

            sinks.extend((event.vertex_a.point, event.vertex_b.point))
            next_event = new_vertex.next_event()

            if next_event is not None:
                events.append(next_event)

        return (Subtree(event.intersection_point, event.distance, sinks), events)

    def handle_split_event(self, event):
        lav = event.vertex.lav

        sinks = [event.vertex.point]
        vertices = []
        x = None  # Next vertex.
        y = None  # Previous vertex.
        norm = event.opposite_edge.norm

        for v in chain.from_iterable(self._lavs):
            if norm == v.edge_prev.norm and event.opposite_edge.p1 == v.edge_prev.p1:
                x = v
                y = x.prev
            elif norm == v.edge_next.norm and event.opposite_edge.p1 == v.edge_next.p1:
                y = v
                x = y.next

            if x:
                x_prev = (y.bisector.v.normalized()).cross(
                    (event.intersection_point - y.point).normalized()
                ) <= EPSILON
                x_next = (x.bisector.v.normalized()).cross(
                    (event.intersection_point - x.point).normalized()
                ) >= -EPSILON

                if x_prev and x_next:
                    break
                else:
                    x = None
                    y = None

        if x is None:
            # Split event failed (equivalent edge event is expected to follow).
            return (None, [])

        v1 = LAVertex(
            event.intersection_point, event.vertex.edge_prev, event.opposite_edge, None, True
        )
        v2 = LAVertex(
            event.intersection_point, event.opposite_edge, event.vertex.edge_next, None, True
        )

        v1.prev = event.vertex.prev
        v1.next = x
        event.vertex.prev.next = v1
        x.prev = v1

        v2.prev = y
        v2.next = event.vertex.next
        event.vertex.next.prev = v2
        y.next = v2

        new_lavs = None
        self._lavs.remove(lav)

        if lav != x.lav:
            # The split event actually merges two lavs.
            self._lavs.remove(x.lav)
            new_lavs = [LAV.from_chain(v1, self)]
        else:
            new_lavs = [LAV.from_chain(v1, self), LAV.from_chain(v2, self)]

        for new_lav in new_lavs:
            if len(new_lav) > 2:
                self._lavs.append(new_lav)
                vertices.append(new_lav.head)
            else:
                # LAV has collapsed into the line.
                sinks.append(new_lav.head.next.point)

                for v in list(new_lav):
                    v.invalidate()

        events = []

        for vertex in vertices:
            next_event = vertex.next_event()

            if next_event is not None:
                events.append(next_event)

        event.vertex.invalidate()

        return (Subtree(event.intersection_point, event.distance, sinks), events)


class EventQueue:
    def __init__(self):
        self.__data = []

    def put(self, item):
        if item is not None:
            heapq.heappush(self.__data, item)

    def put_all(self, iterable):
        for item in iterable:
            heapq.heappush(self.__data, item)

    def get(self):
        return heapq.heappop(self.__data)

    def get_all_equal_distance(self):
        item = heapq.heappop(self.__data)
        equal_distance_list = [item]

        # From top of queue, get all events that have the same distance as the one on top.
        while self.__data and abs(self.__data[0].distance - item.distance) < 0.001:
            queue_top = heapq.heappop(self.__data)
            equal_distance_list.append(queue_top)

        return equal_distance_list

    def empty(self):
        return not self.__data

    def peek(self):
        return self.__data[0]

    def show(self):
        for item in self.__data:
            print(item)


def check_edge_crossing(skeleton):
    # Extract all edges.
    sk_edges = []

    for arc in skeleton:
        p1 = arc.source

        for p2 in arc.sinks:
            sk_edges.append(Edge2(p1, p2))

    combs = combinations(sk_edges, 2)
    intersections_no = 0

    for e in combs:
        # Check for intersection, exclude endpoints.
        denom = ((e[0].p2.x - e[0].p1.x) * (e[1].p2.y - e[1].p1.y)) - (
            (e[0].p2.y - e[0].p1.y) * (e[1].p2.x - e[1].p1.x)
        )

        if not denom:
            continue

        n1 = ((e[0].p1.y - e[1].p1.y) * (e[1].p2.x - e[1].p1.x)) - (
            (e[0].p1.x - e[1].p1.x) * (e[1].p2.y - e[1].p1.y)
        )
        r = n1 / denom
        n2 = ((e[0].p1.y - e[1].p1.y) * (e[0].p2.x - e[0].p1.x)) - (
            (e[0].p1.x - e[1].p1.x) * (e[0].p2.y - e[0].p1.y)
        )
        s = n2 / denom

        if (r <= EPSILON or r >= 1.0 - EPSILON) or (s <= EPSILON or s >= 1.0 - EPSILON):
            # No intersection.
            continue
        else:
            intersections_no += 1

    return intersections_no


def remove_ghosts(skeleton):
    # Remove loops.
    for arc in skeleton:
        if arc.source in arc.sinks:
            arc.sinks.remove(arc.source)

    # Find and resolve parallel or crossed skeleton edges.
    for arc in skeleton:
        source = arc.source

        # Search for nearly parallel edges in all sinks from this node.
        sinks_altered = True

        while sinks_altered:
            sinks_altered = False
            combs = combinations(arc.sinks, 2)

            for pair in combs:
                s0 = pair[0] - source
                s1 = pair[1] - source
                s0m = s0.magnitude
                s1m = s1.magnitude

                if s0m != 0.0 and s1m != 0.0:
                    # Check if this pair of edges is parallel.
                    dot_cosine_abs = abs(s0.dot(s1) / (s0m * s1m) - 1.0)

                    if dot_cosine_abs < PARALLEL:
                        if s0m < s1m:
                            far_sink = pair[1]
                            near_sink = pair[0]
                        else:
                            far_sink = pair[0]
                            near_sink = pair[1]

                        node_index_list = [
                            i for i, node in enumerate(skeleton) if node.source == near_sink
                        ]

                        # Both sinks point to polygon vertices (maybe small triangle).
                        if not node_index_list:
                            break

                        node_index = node_index_list[0]

                        # We have a ghost edge, sinks almost parallel.
                        if dot_cosine_abs < EPSILON:
                            skeleton[node_index].sinks.append(far_sink)
                            arc.sinks.remove(far_sink)
                            arc.sinks.remove(near_sink)
                            sinks_altered = True
                            break
                        else:  # maybe we have a spike that crosses other skeleton edges
                            # Spikes normally get removed with more success as face-spike in polygonize().
                            # Remove it here only, if it produces any crossing.
                            for sink in skeleton[node_index].sinks:
                                if intersect(source, far_sink, near_sink, sink):
                                    skeleton[node_index].sinks.append(far_sink)
                                    arc.sinks.remove(far_sink)
                                    arc.sinks.remove(near_sink)
                                    sinks_altered = True
                                    break


def detect_apses(outer_contour):
    # Compute cross-product between consecutive edges of outer contour
    # Set True for angles a, where sin(a) < 0.5 -> 30°.
    sequence = "".join(
        [
            'L' if abs(p.norm.cross(n.norm)) < 0.5 else 'H'
            for p, n in iter_circular_prev_next(outer_contour)
        ]
    )

    # Special case, see test_306011654_pescara_pattinodromo.
    if all([p == 'L' for p in sequence]):
        return None

    # Match at least 6 low angles in sequence (assume that the first match is longest).
    # Sequence may be circular, like 'LLHHHHHLLLLL'.
    matches = [r for r in APSE_PATTERN.finditer(sequence + sequence)]

    if not matches:
        return None

    centers = []
    N = len(sequence)

    # Circular overlapping pattern must start in first sequence.
    next_start = 0

    for apse in matches:
        s = apse.span()[0]

        if s < N and s >= next_start:
            apse_indices = [i % N for i in range(*apse.span())]
            apse_vertices = [outer_contour[i].p1 for i in apse_indices]
            center, R = fit_circle_3_points(apse_vertices)
            centers.append(center)

    return centers


def find_clusters(skeleton, candidates, contour_vertices, edge_contours, threshold):
    apse_centers = detect_apses(edge_contours[0])
    clusters = []

    while candidates:
        c0 = candidates[0]
        cluster = [c0]
        ref = skeleton[c0]

        for c in candidates[1:]:
            arc = skeleton[c]

            # Use Manhattan distance.
            if abs(ref.source.x - arc.source.x) + abs(ref.source.y - arc.source.y) < threshold:
                cluster.append(c)

        for c in cluster:
            if c in candidates:
                candidates.remove(c)

        if len(cluster) > 1:
            # If cluster is near to an apse center, don't merge any nodes.
            if apse_centers:
                is_apse_cluster = False

                for apse_center in apse_centers:
                    for node in cluster:
                        if (
                            abs(apse_center.x - skeleton[node].source.x)
                            + abs(apse_center.y - skeleton[node].source.y)
                            < 3.0
                        ):
                            is_apse_cluster = True
                            break

                    if is_apse_cluster:
                        break

                if is_apse_cluster:
                    continue

            # Detect sinks in this cluster, that are contour vertices of the footprint.
            contour_sinks_no = 0
            contour_sinks = []

            for node in cluster:
                sinks = skeleton[node].sinks
                contour_sinks.extend([s for s in sinks if s in contour_vertices])
                contour_sinks_no += sum(el in sinks for el in contour_vertices)

            # Less than 2, then we can merge the cluster.
            if contour_sinks_no < 2:
                clusters.append(cluster)
                continue

            # Two or more contour sinks, maybe its an architectural detail, that we shouldn't merge.
            # There are only few sinks, so the minimal distance is computed by brute force.
            min_dist = 3 * threshold
            combs = combinations(contour_sinks, 2)

            for pair in combs:
                min_dist = min((pair[0] - pair[1]).magnitude, min_dist)

            if min_dist > 2 * threshold:
                # Contour sinks too far, so merge.
                clusters.append(cluster)

    return clusters


def merge_cluster(skeleton, cluster):
    nodes_to_merge = cluster.copy()

    # Compute center of gravity as source of merged node.
    # At the same time, collect all sinks of the merged nodes.
    x, y, height = (0.0, 0.0, 0.0)
    merged_sources = []

    for node in cluster:
        x += skeleton[node].source.x
        y += skeleton[node].source.y
        height += skeleton[node].height
        merged_sources.append(skeleton[node].source)

    N = len(cluster)
    new_source = mathutils.Vector((x / N, y / N))
    new_height = height / N

    # Collect all sinks of merged nodes, that point outside the cluster.
    new_sinks = []

    for node in cluster:
        for sink in skeleton[node].sinks:
            if sink not in merged_sources and sink not in new_sinks:
                new_sinks.append(sink)

    # Create the merged node.
    newnode = Subtree(new_source, new_height, new_sinks)

    # Redirect all sinks of nodes outside the cluster, that pointed to
    # one of the clustered nodes, to the new node.
    for arc in skeleton:
        if arc.source not in merged_sources:
            to_remove = []

            for i, sink in enumerate(arc.sinks):
                if sink in merged_sources:
                    if new_source in arc.sinks:
                        to_remove.append(i)
                    else:
                        arc.sinks[i] = new_source

            for i in sorted(to_remove, reverse=True):
                del arc.sinks[i]

    # Remove clustered nodes from skeleton and add the new node.
    for i in sorted(nodes_to_merge, reverse=True):
        del skeleton[i]

    skeleton.append(newnode)


def merge_node_clusters(skeleton, edge_contours):
    # First merge all nodes that have exactly the same source.
    sources = {}
    to_remove = []

    for i, p in enumerate(skeleton):
        source = tuple(i for i in p.source)

        if source in sources:
            source_index = sources[source]

            # Source exists, merge sinks.
            for sink in p.sinks:
                if sink not in skeleton[source_index].sinks:
                    skeleton[source_index].sinks.append(sink)

            to_remove.append(i)
        else:
            sources[source] = i

    for i in reversed(to_remove):
        skeleton.pop(i)

    contour_vertices = [edge.p1 for contour in edge_contours for edge in contour]

    # Merge all clusters that have small distances due to floating-point inaccuracies.
    small_threshold = 0.1
    had_cluster = True

    while had_cluster:
        had_cluster = False

        # Find clusters within short range and short height difference.
        candidates = [c for c in range(len(skeleton))]
        clusters = find_clusters(
            skeleton, candidates, contour_vertices, edge_contours, small_threshold
        )

        # Check if there are cluster candidates.
        if not clusters:
            break

        had_cluster = True

        # Use largest cluster.
        cluster = max(clusters, key=lambda clstr: len(clstr))
        merge_cluster(skeleton, cluster)

    return skeleton


def detect_dormers(slav, edge_contours):
    outer_contour = edge_contours[0]

    def coder(cp):
        if cp > 0.99:
            code = 'L'
        elif cp < -0.99:
            code = 'R'
        else:
            code = '0'

        return code

    sequence = "".join(
        [coder(p.norm.cross(n.norm)) for p, n in iter_circular_prev_next(outer_contour)]
    )
    N = len(sequence)

    # Match a pattern of almost rectangular turns to right, then to left, to left and
    # again to right. Positive lookahead used to find overlapping patterns. Sequence may
    # be circular, like 'LRLL000LL00RL', therefore concatenate two of them.
    matches = [r for r in DORMER_PATTERN.finditer(sequence + sequence)]

    # Circular overlapping pattern must start in first sequence.
    dormer_indices = []
    next_start = 0

    for dormer in matches:
        s = dormer.span()[0]

        if s < N and s >= next_start:
            # Indices of candidate dormer.
            oi = [i % N for i in range(*(s, s + 4))]
            dormer_indices.append(oi)
            next_start = s + 3

    # Filter overlapping dormers.
    to_remove = []

    for oi1, oi2 in zip(dormer_indices, dormer_indices[1:] + dormer_indices[:1]):
        if oi1[3] == oi2[0]:
            to_remove.extend([oi1, oi2])

    for sp in to_remove:
        if sp in dormer_indices:
            dormer_indices.remove(sp)

    # Check if contour consists only of dormers, if yes then skip, can't handle that
    # (special case for test_51340792_yekaterinburg_mashinnaya_35a).
    dormer_verts = set()

    for oi in dormer_indices:
        dormer_verts.update(oi)

    if len(dormer_verts) == len(outer_contour):
        return []

    dormers = []

    for oi in dormer_indices:
        w = outer_contour[oi[1]].length_squared()  # length^2 of base edge
        d1 = outer_contour[oi[0]].length_squared()  # length^2 of side edge
        d2 = outer_contour[oi[2]].length_squared()  # length^2 of side edge
        d = abs(d1 - d2) / (d1 + d2)  # "contrast" of side edges lengths^2
        d3 = outer_contour[(oi[0] + N - 1) % N].length_squared()  # length^2 of previous edge
        d4 = outer_contour[oi[3]].length_squared()  # length^2 of next edge

        fac_left = 0.125 if sequence[(s + N - 1) % N] != 'L' else 1.5
        fac_right = 0.125 if sequence[(s + 4) % N] != 'L' else 1.5

        if w < 100 and d < 0.35 and d3 >= w * fac_left and d4 >= w * fac_right:
            dormers.append((oi, (outer_contour[oi[1]].p1 - outer_contour[oi[1]].p2).magnitude))

    return dormers


def process_dormers(dormers, initial_events):
    dormer_events = []
    dormer_event_indices = []

    for dormer in dormers:
        dormer_indices = dormer[0]
        d_events = [ev for i, ev in enumerate(initial_events) if i in dormer_indices]

        # If all events are valid.
        if all([(d is not None) for d in d_events]):
            if (
                not isinstance(d_events[0], SplitEvent)
                or not isinstance(d_events[1], EdgeEvent)
                or not isinstance(d_events[3], SplitEvent)
            ):
                continue

            ev_prev = d_events[0]
            ev_next = d_events[3]
            v_prev = ev_prev.vertex
            v_next = ev_next.vertex
            p = v_prev.bisector.intersect(v_next.bisector)
            d = dormer[1] / 2.0

            # Process events:                         split1       split2       edge
            dormer_events.append(DormerEvent(d, p, [d_events[0], d_events[3], d_events[1]]))
            dormer_event_indices.extend(dormer_indices)

    remaining_events = [ev for i, ev in enumerate(initial_events) if i not in dormer_event_indices]
    del initial_events[:]

    initial_events.extend(remaining_events)
    initial_events.extend(dormer_events)


def skeletonize(edge_contours):
    """
    skeletonize() computes the straight skeleton of a polygon. It accepts a simple description of the
    contour of a footprint polygon, including those of evetual holes, and returns the nodes and edges
    of its straight skeleton.

    The polygon is expected as a list of contours, where every contour is a list of edges of type Edge2
    (imported from bpyeuclid). The outer contour of the polygon is the first list of in the list of
    contours and is expected in counterclockwise order. In the right-handed coordinate system, seen from
    top, the polygon is on the left of its contour.

    If the footprint has holes, their contours are expected as lists of their edges, following the outer
    contour of the polygon. Their edges are in clockwise order, seen from top, the polygon is on the left
    of the hole's contour.

    Arguments:
    ---------
    edge_contours:   A list of contours of the polygon and eventually its holes, where every contour is a
                    list of edges of type `Edge2` (imported from `bpyeuclid`). It is expected to as:

                    edge_contours = [ polygon_edge,<hole1_edges>, <hole2_edges>, ...]

                    polygon_egdes is a list of the edges of the outer polygon contour in counterclockwise
                    order. <hole_edges> is an optional list of the edges of a hole contour in clockwise order.

    Output:
    ------
    return:         A list of subtrees (of type Subtree) of the straight skeleton. A Subtree contains the
                    attributes (source, height, sinks), where source is the node vertex, height is its
                    distance to the nearest polygon edge, and sinks is a list of vertices connected to the
                    node. All vertices are of type mathutils.Vector with two dimension x and y.
    """
    slav = SLAV(edge_contours)
    dormers = detect_dormers(slav, edge_contours)
    initial_events = []

    for lav in slav:
        for vertex in lav:
            initial_events.append(vertex.next_event())

    if dormers:
        process_dormers(dormers, initial_events)

    output = []
    prioque = EventQueue()

    for ev in initial_events:
        if ev:
            prioque.put(ev)

    while not (prioque.empty() or slav.empty()):
        top_event_list = prioque.get_all_equal_distance()

        for i in top_event_list:
            if isinstance(i, EdgeEvent):
                if not i.vertex_a.is_valid or not i.vertex_b.is_valid:
                    continue
                (arc, events) = slav.handle_edge_event(i)
            elif isinstance(i, SplitEvent):
                if not i.vertex.is_valid:
                    continue
                (arc, events) = slav.handle_split_event(i)
            elif isinstance(i, DormerEvent):
                if not i.event_list[0].vertex.is_valid or not i.event_list[1].vertex.is_valid:
                    continue
                (arc, events) = slav.handle_dormer_event(i)

            prioque.put_all(events)

            if arc is not None:
                if isinstance(arc, list):
                    output.extend(arc)
                else:
                    output.append(arc)

    output = merge_node_clusters(output, edge_contours)
    remove_ghosts(output)

    return output


def polygonize(
    verts,
    first_vertex_index,
    vertices_no,
    holes_info=None,
    height=0.0,
    tan=0.0,
    faces=None,
    unit_vectors=None,
):
    """
    polygonize() computes the faces of a hipped roof from a footprint polygon of a building, skeletonized
    by a straight skeleton. It accepts a simple description of the vertices of the footprint polygon,
    including those of evetual holes, and returns a list of polygon faces.

    The polygon is expected as a list of vertices in counterclockwise order. In a right-handed coordinate
    system, seen from top, the polygon is on the left of its contour. Holes are expected as lists of vertices
    in clockwise order. Seen from top, the polygon is on the left of the hole's contour.

    Arguments:
    ----------
    verts:              A list of vertices. Vertices that define the outer contour of the footprint polygon are
                        located in a continuous block of the verts list, starting at the index first_vertex_index.
                        Each vertex is an instance of `mathutils.Vector` with 3 coordinates x, y and z. The
                        z-coordinate must be the same for all vertices of the polygon.

                        The outer contour of the footprint polygon contains `vertices_no` vertices in counterclockwise
                        order, in its block in `verts`.

                        Vertices that define eventual holes are also located in `verts`. Every hole takes its continuous
                        block. The start index and the length of every hole block are described by the argument
                        `holes_info`. See there.

                        The list of vertices verts gets extended by `polygonize()`. The new nodes of the straight
                        skeleton are appended at the end of the list.

    first_vertex_index: The first index of vertices of the polygon index in the verts list that defines the footprint polygon.

    vertices_no:        The first index of the vertices in the verts list of the polygon, that defines the outer
                        contour of the footprint.

    holes_info:          If the footprint polygon contains holes, their position and length in the verts list are
                        described by this argument. `holes_info` is a list of tuples, one for every hole. The first
                        element in every tuple is the start index of the hole's vertices in `verts` and the second
                        element is the number of its vertices.

                        The default value of holes_info is None, which means that there are no holes.

    height: 	        The maximum height of the hipped roof to be generated. If both `height` and `tan` are equal
                        to zero, flat faces are generated. `height` takes precedence over `tan` if both have a non-zero
                        value. The default value of `height` is 0.0.

    tan:                In many cases it's desirable to deal with the roof pitch angle instead of the maximum roof
                        height. The tangent `tan` of the roof pitch angle can be supplied for that case. If both `height`
                        and `tan` are equal to zero, flat faces are generated. `height` takes precedence over `tan` if
                        both have a non-zero value. The default value of `tan` is 0.0.

    faces:              An already existing Python list of faces. Every face in this list is itself a list of
                        indices of the face-vertices in the verts list. If this argument is None (its default value),
                        a new list with the new faces created by the straight skeleton is created and returned by
                        polygonize(), else faces is extended by the new list.

    unit_vectors:       A Python list of unit vectors along the polygon edges (including holes if they are present).
                        These vectors are of type `mathutils.Vector` with three dimensions. The direction of the vectors
                        corresponds to order of the vertices in the polygon and its holes. The order of the unit
                        vectors in the unit_vectors list corresponds to the order of vertices in the input Python list
                        verts.

                        The list `unit_vectors` (if given) gets used inside polygonize() function instead of calculating
                        it once more. If this argument is None (its default value), the unit vectors get calculated
                        inside polygonize().

    Output:
    ------
    verts:              The list of the vertices `verts` gets extended at its end by the vertices of the straight skeleton.

    return:             A list of the faces created by the straight skeleton. Every face in this list is a list of
                        indices of the face-vertices in the verts list. The order of vertices of the faces is
                        counterclockwise, as the order of vertices in the input Python list `verts`. The first edge of
                        a face is always an edge of the polygon or its holes.

                        If a list of faces has been given in the argument faces, it gets extended at its end by the
                        new list.
    """
    # Assume that all vertices of polygon and holes have the same z-value.
    z_base = verts[first_vertex_index][2]

    # Compute center of gravity of polygon.
    center = mathutils.Vector((0.0, 0.0, 0.0))

    for i in range(first_vertex_index, first_vertex_index + vertices_no):
        center += verts[i]

    center /= vertices_no
    center[2] = 0.0

    # Create 2D edges as list and as contours for skeletonization and graph construction.
    last_u_index = vertices_no - 1
    last_vert_index = first_vertex_index + last_u_index

    if unit_vectors:
        edges2D = [
            Edge2(index, index + 1, unit_vectors[u_index], verts, center)
            for index, u_index in zip(
                range(first_vertex_index, last_vert_index), range(last_u_index)
            )
        ]
        edges2D.append(
            Edge2(last_vert_index, first_vertex_index, unit_vectors[last_u_index], verts, center)
        )
    else:
        edges2D = [
            Edge2(index, index + 1, None, verts, center)
            for index in range(first_vertex_index, last_vert_index)
        ]
        edges2D.append(Edge2(last_vert_index, first_vertex_index, None, verts, center))

    edge_contours = [edges2D.copy()]
    u_index = vertices_no

    if holes_info:
        for first_vertex_index_hole, vertices_no_hole in holes_info:
            last_vertex_index_hole = first_vertex_index_hole + vertices_no_hole - 1

            if unit_vectors:
                last_u_index = u_index + vertices_no_hole - 1
                hole_edges = [
                    Edge2(index, index + 1, unit_vectors[u_index], verts, center)
                    for index, u_index in zip(
                        range(first_vertex_index_hole, last_vertex_index_hole),
                        range(u_index, last_u_index),
                    )
                ]
                hole_edges.append(
                    Edge2(
                        last_vertex_index_hole,
                        first_vertex_index_hole,
                        unit_vectors[last_u_index],
                        verts,
                        center,
                    )
                )
            else:
                hole_edges = [
                    Edge2(index, index + 1, None, verts, center)
                    for index in range(first_vertex_index_hole, last_vertex_index_hole)
                ]
                hole_edges.append(
                    Edge2(last_vertex_index_hole, first_vertex_index_hole, None, verts, center)
                )
            edges2D.extend(hole_edges)
            edge_contours.append(hole_edges)
            u_index += vertices_no_hole

    # Compute skeleton.
    skeleton = skeletonize(edge_contours)

    # Eventual debug output of skeleton.
    if 'skeleton' in debug_outputs:
        debug_outputs['skeleton'] = skeleton

    # Compute skeleton node heights and append nodes to original verts list.
    # See also issue #4 at https://github.com/prochitecture/bpypolyskel.
    if height:
        max_skel_height = max(arc.height for arc in skeleton)
        tan_alpha = height / max_skel_height
    else:
        tan_alpha = tan

    skeleton_nodes3D = []

    for arc in skeleton:
        node = mathutils.Vector((arc.source.x, arc.source.y, arc.height * tan_alpha + z_base))
        skeleton_nodes3D.append(node + center)

    first_skel_index = len(verts)
    verts.extend(skeleton_nodes3D)

    # Instantiate the graph for faces.
    graph = Poly2FacesGraph()

    # Add polygon and hole indices to graph using indices in verts.
    for edge in iter_circular_prev_next(
        range(first_vertex_index, first_vertex_index + vertices_no)
    ):
        graph.add_edge(edge)

    if holes_info:
        for first_vertex_index_hole, vertices_no_hole in holes_info:
            for edge in iter_circular_prev_next(
                range(first_vertex_index_hole, first_vertex_index_hole + vertices_no_hole)
            ):
                graph.add_edge(edge)

    # Add skeleton edges to graph using indices in verts.
    for index, arc in enumerate(skeleton):
        a_index = index + first_skel_index

        for sink in arc.sinks:
            # First search in input edges.
            edge = [edge for edge in edges2D if edge.p1 == sink]

            if edge:
                s_index = edge[0].i1

            # Then it should be a skeleton node.
            else:
                skel_index = [index for index, arc in enumerate(skeleton) if arc.source == sink]

                if skel_index:
                    s_index = skel_index[0] + first_skel_index
                else:
                    # Error.
                    s_index = -1

            graph.add_edge((a_index, s_index))

    # Generate clockwise circular embedding.
    embedding = graph.circular_embedding(verts, 'CCW')

    # Compute list of faces, the vertex indices are still related to verts2D.
    faces3D = graph.faces(embedding, first_skel_index)

    # Find and remove spikes in faces.
    had_spikes = True

    while had_spikes:
        had_spikes = False

        # Find spike.
        for face in faces3D:
            # A triangle is not considered as spike.
            if len(face) <= 3:
                continue

            for prev, this, _next in iter_circular_prev_this_next(face):
                # 'verts' are 3D vectors.
                s0 = verts[this] - verts[prev]
                s1 = verts[_next] - verts[this]

                # Need 2D-vectors.
                s0 = s0.xy
                s1 = s1.xy

                s0m = s0.magnitude
                s1m = s1.magnitude

                if s0m and s1m:
                    dot_cosine = s0.dot(s1) / (s0m * s1m)
                else:
                    continue

                cross_sine = s0.cross(s1)

                # Spike edge to left.
                if abs(dot_cosine + 1.0) < PARALLEL and cross_sine > -EPSILON:
                    # The spike's peak is at 'this'.
                    had_spikes = True
                    break
                else:
                    continue

            if not had_spikes:
                # Try next face.
                continue

            # Find faces adjacent to spike.
            # On right side it must have adjacent vertices in the order 'this' -> 'prev';
            # On left side it must have adjacent vertices in the order '_next' -> 'this.
            right_index, left_index = (None, None)

            for i, f in enumerate(faces3D):
                if [p for p, n in iter_circular_prev_next(f) if p == this and n == prev]:
                    right_index = i

                if [p for p, n in iter_circular_prev_next(f) if p == _next and n == this]:
                    left_index = i

            # Part of spike is original polygon and cant get removed.
            if right_index is None or left_index is None:
                had_spikes = False
                continue

            # Single line into a face, but not separating it.
            if right_index == left_index:
                common_face = faces3D[right_index]

                # Remove the spike vertice and one of its neighbors.
                common_face.remove(this)
                common_face.remove(prev)

                if this in face:
                    face.remove(this)

                # That's it for this face.
                break

            # Rotate right face so that 'prev' is in first place.
            right_face = faces3D[right_index]
            rot_index = next(x[0] for x in enumerate(right_face) if x[1] == prev)
            right_face = right_face[rot_index:] + right_face[:rot_index]

            # Rotate left face so that 'this' is in first place.
            left_face = faces3D[left_index]
            rot_index = next(x[0] for x in enumerate(left_face) if x[1] == this)
            left_face = left_face[rot_index:] + left_face[:rot_index]

            merged_face = right_face + left_face[1:]

            # Rotate edge list so that edge of original polygon is first edge.
            next_orig_index = next(
                x[0]
                for x in enumerate(merged_face)
                if x[0] < first_skel_index and x[1] < first_skel_index
            )
            merged_face = merged_face[next_orig_index:] + merged_face[:next_orig_index]

            if merged_face == face:  # no change, will result in endless loop
                raise Exception('Endless loop in spike removal')

            # Remove the spike.
            face.remove(this)

            for i in sorted([right_index, left_index], reverse=True):
                del faces3D[i]

            faces3D.append(merged_face)

            # Break looping through faces and restart main while loop,
            # because it is possible that new spikes have been generated.
            break

    # Fix adjacent parallel edges in faces.
    counts = Counter(chain.from_iterable(faces3D))

    for face in faces3D:
        # A triangle can't have parallel edges.
        if len(face) > 3:
            vertices_to_remove = []

            for prev, this, _next in iter_circular_prev_this_next(face):
                # Can eventually remove vertice, if it appears only in
                # two adjacent faces, otherwise its a node.
                # But do not remove original polygon vertices.
                if counts[this] < 3 and this >= first_skel_index:
                    s0 = verts[this] - verts[prev]
                    s1 = verts[_next] - verts[this]

                    # Need 2D-vector.
                    s0 = mathutils.Vector((s0[0], s0[1]))
                    s1 = mathutils.Vector((s1[0], s1[1]))

                    s0m = s0.magnitude
                    s1m = s1.magnitude

                    if s0m != 0.0 and s1m != 0.0:
                        dot_cosine = s0.dot(s1) / (s0m * s1m)

                        # Found adjacent parallel edges.
                        if abs(dot_cosine - 1.0) < PARALLEL:
                            vertices_to_remove.append(this)
                    else:
                        # Duplicate vertex.
                        if this not in vertices_to_remove:
                            vertices_to_remove.append(this)

            for item in vertices_to_remove:
                face.remove(item)

        # Remove one of adjacent identical vertices.
        vertices_to_remove = []

        for prev, _next in iter_circular_prev_next(face):
            if prev == _next:
                vertices_to_remove.append(prev)

        for item in vertices_to_remove:
            face.remove(item)

    if faces is None:
        return faces3D

    faces.extend(faces3D)
    return faces
