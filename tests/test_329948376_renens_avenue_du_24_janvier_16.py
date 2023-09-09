import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((8.407790184020996, -6.723691463470459, 0.0)),
    Vector((12.190521240234375, -2.181849718093872, 0.0)),
    Vector((14.29629898071289, -3.3618316650390625, 0.0)),
    Vector((19.174076080322266, -14.638483047485352, 0.0)),
    Vector((28.92955780029297, -10.853581428527832, 0.0)),
    Vector((23.048643112182617, 2.437940835952759, 0.0)),
    Vector((21.754545211791992, 4.007540702819824, 0.0)),
    Vector((19.901458740234375, 5.610535144805908, 0.0)),
    Vector((18.293411254882812, 6.734857082366943, 0.0)),
    Vector((15.291727066040039, 7.903703212738037, 0.0)),
    Vector((12.190500259399414, 8.07067584991455, 0.0)),
    Vector((8.698750495910645, 7.391620635986328, 0.0)),
    Vector((5.582210063934326, 5.510317325592041, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((8.407790184020996, -6.723691463470459, 17.963478088378906)),
    Vector((12.190521240234375, -2.181849718093872, 17.963478088378906)),
    Vector((14.29629898071289, -3.3618316650390625, 17.963478088378906)),
    Vector((19.174076080322266, -14.638483047485352, 17.963478088378906)),
    Vector((28.92955780029297, -10.853581428527832, 17.963478088378906)),
    Vector((23.048643112182617, 2.437940835952759, 17.963478088378906)),
    Vector((21.754545211791992, 4.007540702819824, 17.963478088378906)),
    Vector((19.901458740234375, 5.610535144805908, 17.963478088378906)),
    Vector((18.293411254882812, 6.734857082366943, 17.963478088378906)),
    Vector((15.291727066040039, 7.903703212738037, 17.963478088378906)),
    Vector((12.190500259399414, 8.07067584991455, 17.963478088378906)),
    Vector((8.698750495910645, 7.391620635986328, 17.963478088378906)),
    Vector((5.582210063934326, 5.510317325592041, 17.963478088378906)),
    Vector((0.0, 0.0, 17.963478088378906))
]
unitVectors = [
    Vector((0.6399709582328796, 0.7683990001678467, 0.0)),
    Vector((0.8723741769790649, -0.4888387620449066, 0.0)),
    Vector((0.3970062732696533, -0.9178159236907959, 0.0)),
    Vector((0.932291567325592, 0.3617075979709625, 0.0)),
    Vector((-0.4046195149421692, 0.9144851565361023, 0.0)),
    Vector((-0.6361424922943115, 0.7715715765953064, 0.0)),
    Vector((-0.7562975883483887, 0.6542278528213501, 0.0)),
    Vector((-0.8195458054542542, 0.5730137228965759, 0.0)),
    Vector((-0.9318447113037109, 0.3628573417663574, 0.0)),
    Vector((-0.9985536932945251, 0.05376296490430832, 0.0)),
    Vector((-0.9816099405288696, -0.19089780747890472, 0.0)),
    Vector((-0.8561108112335205, -0.5167922973632812, 0.0)),
    Vector((-0.711674690246582, -0.7025091052055359, 0.0)),
    Vector((0.7809839248657227, -0.6245511174201965, 0.0))
]
holesInfo = None
firstVertIndex = 14
numPolygonVerts = 14
faces = []

bpypolyskel.debugOutputs["skeleton"] = 1


@pytest.mark.dependency()
@pytest.mark.timeout(10)
def test_polygonize():
    global faces
    faces = bpypolyskel.polygonize(verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, unitVectors)


@pytest.mark.dependency(depends=["test_polygonize"])
def test_numVertsInFace():
    for face in faces:
        assert len(face) >= 3


@pytest.mark.dependency(depends=["test_polygonize"])
def test_duplication():
    for face in faces:
        assert len(face) == len(set(face))


@pytest.mark.dependency(depends=["test_polygonize"])
def test_edgeCrossing():
    assert not bpypolyskel.check_edge_crossing(bpypolyskel.debugOutputs["skeleton"])
