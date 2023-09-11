import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-0.09972011297941208, -9.874038696289062, 0.0)),
    Vector((66.52579498291016, -10.541441917419434, 0.0)),
    Vector((66.66270446777344, 2.7835028171539307, 0.0)),
    Vector((54.77734375, 2.9057867527008057, 0.0)),
    Vector((54.74622344970703, -0.5451177954673767, 0.0)),
    Vector((24.269329071044922, -0.23370260000228882, 0.0)),
    Vector((24.46242904663086, 18.701744079589844, 0.0)),
    Vector((8.557175636291504, 18.85753059387207, 0.0)),
    Vector((8.370238304138184, -0.07791551202535629, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((-0.09972011297941208, -9.874038696289062, 8.725227355957031)),
    Vector((66.52579498291016, -10.541441917419434, 8.725227355957031)),
    Vector((66.66270446777344, 2.7835028171539307, 8.725227355957031)),
    Vector((54.77734375, 2.9057867527008057, 8.725227355957031)),
    Vector((54.74622344970703, -0.5451177954673767, 8.725227355957031)),
    Vector((24.269329071044922, -0.23370260000228882, 8.725227355957031)),
    Vector((24.46242904663086, 18.701744079589844, 8.725227355957031)),
    Vector((8.557175636291504, 18.85753059387207, 8.725227355957031)),
    Vector((8.370238304138184, -0.07791551202535629, 8.725227355957031)),
    Vector((0.0, 0.0, 8.725227355957031))
]
unitVectors = [
    Vector((0.9999498128890991, -0.010016728192567825, 0.0)),
    Vector((0.010274133644998074, 0.999947190284729, 0.0)),
    Vector((-0.9999471306800842, 0.010288074612617493, 0.0)),
    Vector((-0.009017645381391048, -0.9999592900276184, 0.0)),
    Vector((-0.9999477863311768, 0.010217541828751564, 0.0)),
    Vector((0.010197275318205357, 0.9999480843544006, 0.0)),
    Vector((-0.9999520778656006, 0.009794188663363457, 0.0)),
    Vector((-0.00987186748534441, -0.9999512434005737, 0.0)),
    Vector((-0.9999567270278931, 0.009308233857154846, 0.0)),
    Vector((-0.010098707862198353, -0.999949038028717, 0.0))
]
holesInfo = None
firstVertIndex = 10
numPolygonVerts = 10
faces = []

bpypolyskel.debug_outputs["skeleton"] = 1


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
    assert not bpypolyskel.check_edge_crossing(bpypolyskel.debug_outputs["skeleton"])
