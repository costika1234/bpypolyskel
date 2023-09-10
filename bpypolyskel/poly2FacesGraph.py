from collections import defaultdict
from functools import cmp_to_key

EPSILON = 0.00001


# Adapted from https://stackoverflow.com/questions/16542042/fastest-way-to-sort-vectors-by-angle-without-actually-computing-that-angle
# Input:  d: difference vector.
# Output: a number from the range [0 .. 4] which is monotonic
#         in the angle this vector makes against the x axis.
def pseudo_angle(d):
    # -1..1 increasing with x.
    p = d[0] / (abs(d[0]) + abs(d[1]))

    if d[1] < 0:
        # 2..4 increasing with x.
        return 3 + p

    # 0..2 decreasing with x.
    return 1 - p


def compare_angles(v_list, p1, p2, center):
    a1 = pseudo_angle(v_list[p1] - v_list[center])
    a2 = pseudo_angle(v_list[p2] - v_list[center])

    return 1 if a1 < a2 else -1


class Poly2FacesGraph:
    def __init__(self):
        self.g_dict = {}

    def add_vertex(self, vertex):
        # If vertex not yet known, add empty list.
        if vertex not in self.g_dict:
            self.g_dict[vertex] = []

    def add_edge(self, edge):
        # Edge of type set, tuple or list.
        # Loops are not allowed, but not tested.
        edge = set(edge)

        # Exclude loops.
        if len(edge) == 2:
            vertex1 = edge.pop()
            vertex2 = edge.pop()
            self.add_vertex(vertex1)
            self.add_vertex(vertex2)
            self.g_dict[vertex1].append(vertex2)
            self.g_dict[vertex2].append(vertex1)

    def edges(self):
        # Returns the edges of the graph.
        edges = []

        for vertex in self.g_dict:
            for neighbour in self.g_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append((vertex, neighbour))

        return edges

    def circular_embedding(self, v_list, direction='CCW'):
        embedding = defaultdict(list)

        for vertex in self.g_dict:
            neighbors = self.g_dict[vertex]
            ordering = sorted(
                neighbors, key=cmp_to_key(lambda a, b: compare_angles(v_list, a, b, vertex))
            )

            if direction == 'CCW':
                embedding[vertex] = ordering
            elif direction == 'CW':
                embedding[vertex] = ordering[::-1]

        return embedding

    def faces(self, embedding, poly_vertices_no):
        # Adapted from SAGE's trace_faces.

        # Establish set of possible edges.
        edge_set = set()

        for edge in self.edges():
            edge_set = edge_set.union({(edge[0], edge[1]), (edge[1], edge[0])})

        # Storage for face paths.
        faces, path, face_id = [], [], 0

        for edge in edge_set:
            path.append(edge)
            edge_set -= {edge}

            # (Only one iteration).
            break

        # Trace faces.
        while len(edge_set) > 0:
            neighbors = embedding[path[-1][-1]]
            next_node = neighbors[(neighbors.index(path[-1][-2]) + 1) % (len(neighbors))]
            tup = (path[-1][-1], next_node)

            if tup == path[0]:
                # Convert edge list in vertices list.
                vert_list = [e[0] for e in path]
                faces.append(path)
                face_id += 1
                path = []

                for edge in edge_set:
                    path.append(edge)
                    edge_set -= {edge}

                    # (Only one iteration).
                    break
            else:
                if tup in path:
                    raise Exception('Endless loop catched in Poly2FacesGraph faces().')

                path.append(tup)
                edge_set -= {tup}

        if len(path) != 0:
            # Convert edge list in vertices list.
            vert_list = [e[0] for e in path]
            faces.append(path)

        final_faces = []

        for face in faces:
            # Rotate edge list so that edge of original polygon is first edge.
            orig_edges = [
                x[0]
                for x in enumerate(face)
                if x[1][0] < poly_vertices_no and x[1][1] < poly_vertices_no
            ]

            # If no result: face is floating without polygon contour edges.
            if orig_edges:
                next_orig_index = next(
                    x[0]
                    for x in enumerate(face)
                    if x[1][0] < poly_vertices_no and x[1][1] < poly_vertices_no
                )
                face = face[next_orig_index:] + face[:next_orig_index]

            # Convert edge list in vertices list.
            vert_list = [e[0] for e in face]

            # Exclude polygon and holes.
            if any(i >= poly_vertices_no for i in vert_list):
                final_faces.append(vert_list)

        return final_faces
