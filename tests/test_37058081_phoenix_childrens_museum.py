import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-26.210872650146484, 3.558067692210898e-05, 0.0)),
    Vector((-26.210899353027344, -10.308149337768555, 0.0)),
    Vector((-23.41519546508789, -10.30815601348877, 0.0)),
    Vector((-23.41524887084961, -32.44960403442383, 0.0)),
    Vector((-28.142881393432617, -32.449588775634766, 0.0)),
    Vector((-28.1429386138916, -52.008426666259766, 0.0)),
    Vector((-23.41529655456543, -52.00843811035156, 0.0)),
    Vector((-23.41535186767578, -75.44119262695312, 0.0)),
    Vector((-25.783823013305664, -75.4411849975586, 0.0)),
    Vector((-25.783851623535156, -86.25030517578125, 0.0)),
    Vector((-0.6501691937446594, -86.2503433227539, 0.0)),
    Vector((-0.6501691341400146, -85.22620391845703, 0.0)),
    Vector((15.251110076904297, -85.22618865966797, 0.0)),
    Vector((15.251082420349121, -67.71563720703125, 0.0)),
    Vector((-10.746347427368164, -67.71563720703125, 0.0)),
    Vector((-10.746334075927734, -56.127281188964844, 0.0)),
    Vector((12.455345153808594, -56.12727737426758, 0.0)),
    Vector((12.45531177520752, -30.64624786376953, 0.0)),
    Vector((-10.746305465698242, -30.646249771118164, 0.0)),
    Vector((-10.746294975280762, -21.11730194091797, 0.0)),
    Vector((11.600797653198242, -21.117300033569336, 0.0)),
    Vector((11.600775718688965, -2.5826051235198975, 0.0)),
    Vector((0.0, -2.5826122760772705, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((-26.210872650146484, 3.558067692210898e-05, 15.140739440917969)),
    Vector((-26.210899353027344, -10.308149337768555, 15.140739440917969)),
    Vector((-23.41519546508789, -10.30815601348877, 15.140739440917969)),
    Vector((-23.41524887084961, -32.44960403442383, 15.140739440917969)),
    Vector((-28.142881393432617, -32.449588775634766, 15.140739440917969)),
    Vector((-28.1429386138916, -52.008426666259766, 15.140739440917969)),
    Vector((-23.41529655456543, -52.00843811035156, 15.140739440917969)),
    Vector((-23.41535186767578, -75.44119262695312, 15.140739440917969)),
    Vector((-25.783823013305664, -75.4411849975586, 15.140739440917969)),
    Vector((-25.783851623535156, -86.25030517578125, 15.140739440917969)),
    Vector((-0.6501691937446594, -86.2503433227539, 15.140739440917969)),
    Vector((-0.6501691341400146, -85.22620391845703, 15.140739440917969)),
    Vector((15.251110076904297, -85.22618865966797, 15.140739440917969)),
    Vector((15.251082420349121, -67.71563720703125, 15.140739440917969)),
    Vector((-10.746347427368164, -67.71563720703125, 15.140739440917969)),
    Vector((-10.746334075927734, -56.127281188964844, 15.140739440917969)),
    Vector((12.455345153808594, -56.12727737426758, 15.140739440917969)),
    Vector((12.45531177520752, -30.64624786376953, 15.140739440917969)),
    Vector((-10.746305465698242, -30.646249771118164, 15.140739440917969)),
    Vector((-10.746294975280762, -21.11730194091797, 15.140739440917969)),
    Vector((11.600797653198242, -21.117300033569336, 15.140739440917969)),
    Vector((11.600775718688965, -2.5826051235198975, 15.140739440917969)),
    Vector((0.0, -2.5826122760772705, 15.140739440917969)),
    Vector((0.0, 0.0, 15.140739440917969)),
]
unitVectors = [
    Vector((-2.5904541871568654e-06, -1.0, 0.0)),
    Vector((1.0, -2.387849463048042e-06, 0.0)),
    Vector((-2.4120265607052715e-06, -1.0, 0.0)),
    Vector((-0.9999999403953552, 3.227575007258565e-06, 0.0)),
    Vector((-2.925555008914671e-06, -0.9999999403953552, 0.0)),
    Vector((1.0, -2.4206765374401584e-06, 0.0)),
    Vector((-2.360504140597186e-06, -1.0, 0.0)),
    Vector((-1.0, 3.22123173646105e-06, 0.0)),
    Vector((-2.6468601390661206e-06, -1.0, 0.0)),
    Vector((1.0, -1.5177630530160968e-06, 0.0)),
    Vector((5.819973836196368e-08, 1.0, 0.0)),
    Vector((0.9999999403953552, 9.59595013227954e-07, 0.0)),
    Vector((-1.5794222463227925e-06, 1.0, 0.0)),
    Vector((-1.0, 0.0, 0.0)),
    Vector((1.152142772298248e-06, 1.0, 0.0)),
    Vector((0.9999999403953552, 1.6441470052086515e-07, 0.0)),
    Vector((-1.309939307247987e-06, 1.0, 0.0)),
    Vector((-1.0, -8.220757052868066e-08, 0.0)),
    Vector((1.10089990812412e-06, 1.0, 0.0)),
    Vector((1.0, 8.535108975138428e-08, 0.0)),
    Vector((-1.18342973109975e-06, 1.0, 0.0)),
    Vector((-1.0, -6.165585091366665e-07, 0.0)),
    Vector((0.0, 1.0, 0.0)),
    Vector((-1.0, 1.357477799501794e-06, 0.0)),
]
holesInfo = None
firstVertIndex = 24
numPolygonVerts = 24
faces = []

bpypolyskel.debug_outputs["skeleton"] = 1


@pytest.mark.dependency()
@pytest.mark.timeout(10)
def test_polygonize():
    global faces
    faces = bpypolyskel.polygonize(
        verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, unitVectors
    )


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
    assert not bpypolyskel.check_edge_crossing(bpypolyskel.debug_outputs["skeleton"])
