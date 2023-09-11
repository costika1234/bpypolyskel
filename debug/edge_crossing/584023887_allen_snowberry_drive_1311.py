from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-0.19577747583389282, -11.18760871887207, 0.0)),
    Vector((3.0671803951263428, -11.243268013000488, 0.0)),
    Vector((3.011245012283325, -14.293421745300293, 0.0)),
    Vector((0.8297240734100342, -17.0764102935791, 0.0)),
    Vector((0.7178516387939453, -23.399356842041016, 0.0)),
    Vector((2.8527610301971436, -23.43275260925293, 0.0)),
    Vector((2.806148052215576, -26.148948669433594, 0.0)),
    Vector((10.450803756713867, -26.271394729614258, 0.0)),
    Vector((10.711824417114258, -10.80911636352539, 0.0)),
    Vector((9.481223106384277, -10.786853790283203, 0.0)),
    Vector((9.658344268798828, -0.16697447001934052, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((-0.19577747583389282, -11.18760871887207, 5.300000190734863)),
    Vector((3.0671803951263428, -11.243268013000488, 5.300000190734863)),
    Vector((3.011245012283325, -14.293421745300293, 5.300000190734863)),
    Vector((0.8297240734100342, -17.0764102935791, 5.300000190734863)),
    Vector((0.7178516387939453, -23.399356842041016, 5.300000190734863)),
    Vector((2.8527610301971436, -23.43275260925293, 5.300000190734863)),
    Vector((2.806148052215576, -26.148948669433594, 5.300000190734863)),
    Vector((10.450803756713867, -26.271394729614258, 5.300000190734863)),
    Vector((10.711824417114258, -10.80911636352539, 5.300000190734863)),
    Vector((9.481223106384277, -10.786853790283203, 5.300000190734863)),
    Vector((9.658344268798828, -0.16697447001934052, 5.300000190734863)),
    Vector((0.0, 0.0, 5.300000190734863))
]
unitVectors = [
    Vector((0.999854564666748, -0.017055446282029152, 0.0)),
    Vector((-0.018335461616516113, -0.9998318552970886, 0.0)),
    Vector((-0.6169271469116211, -0.7870202660560608, 0.0)),
    Vector((-0.017690317705273628, -0.9998435378074646, 0.0)),
    Vector((0.9998777508735657, -0.015640797093510628, 0.0)),
    Vector((-0.017158597707748413, -0.9998527765274048, 0.0)),
    Vector((0.9998718500137329, -0.016015157103538513, 0.0)),
    Vector((0.016878722235560417, 0.9998576045036316, 0.0)),
    Vector((-0.9998364448547363, 0.018087849020957947, 0.0)),
    Vector((0.01667594537138939, 0.999860942363739, 0.0)),
    Vector((-0.9998506307601929, 0.01728552207350731, 0.0)),
    Vector((-0.017496813088655472, -0.999846875667572, 0.0))
]
holesInfo = None
firstVertIndex = 12
numPolygonVerts = 12

bpypolyskel.debug_outputs["skeleton"] = 1


faces = bpypolyskel.polygonize(verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, unitVectors)


# the number of vertices in a face
for face in faces:
    assert len(face) >= 3


# duplications of vertex indices
for face in faces:
    assert len(face) == len(set(face))


# edge crossing
assert not bpypolyskel.check_edge_crossing(bpypolyskel.debug_outputs["skeleton"])
