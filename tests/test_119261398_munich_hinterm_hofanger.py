import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, -12.990984916687012, 0.0)),
    Vector((-0.4613610506057739, -12.990984916687012, 0.0)),
    Vector((-0.46136215329170227, -26.68328285217285, 0.0)),
    Vector((11.928443908691406, -26.683269500732422, 0.0)),
    Vector((11.928415298461914, -12.990972518920898, 0.0)),
    Vector((9.614169120788574, -12.990976333618164, 0.0)),
    Vector((9.614147186279297, 8.062602319114376e-06, 0.0)),
    Vector((11.228907585144043, 1.0998449397447985e-05, 0.0)),
    Vector((11.228879928588867, 14.026267051696777, 0.0)),
    Vector((-1.0417780876159668, 14.02625560760498, 0.0)),
    Vector((-1.0417805910110474, 9.488746854913188e-08, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((0.0, -12.990984916687012, 14.513432502746582)),
    Vector((-0.4613610506057739, -12.990984916687012, 14.513432502746582)),
    Vector((-0.46136215329170227, -26.68328285217285, 14.513432502746582)),
    Vector((11.928443908691406, -26.683269500732422, 14.513432502746582)),
    Vector((11.928415298461914, -12.990972518920898, 14.513432502746582)),
    Vector((9.614169120788574, -12.990976333618164, 14.513432502746582)),
    Vector((9.614147186279297, 8.062602319114376e-06, 14.513432502746582)),
    Vector((11.228907585144043, 1.0998449397447985e-05, 14.513432502746582)),
    Vector((11.228879928588867, 14.026267051696777, 14.513432502746582)),
    Vector((-1.0417780876159668, 14.02625560760498, 14.513432502746582)),
    Vector((-1.0417805910110474, 9.488746854913188e-08, 14.513432502746582)),
    Vector((0.0, 0.0, 14.513432502746582))
]
unitVectors = [
    Vector((-1.0, 0.0, 0.0)),
    Vector((-8.053329736412707e-08, -0.9999999403953552, 0.0)),
    Vector((1.0, 1.0776150247693295e-06, 0.0)),
    Vector((-2.0895126908726525e-06, 1.0, 0.0)),
    Vector((-1.0, -1.6483541003253777e-06, 0.0)),
    Vector((-1.6884409888007212e-06, 1.0, 0.0)),
    Vector((0.9999999403953552, 1.8181316363552469e-06, 0.0)),
    Vector((-1.971770416275831e-06, 1.0, 0.0)),
    Vector((-1.0, -9.326387271357817e-07, 0.0)),
    Vector((-1.7847921185421e-07, -1.0, 0.0)),
    Vector((1.0, -9.108200771379416e-08, 0.0)),
    Vector((0.0, -1.0, 0.0))
]
holesInfo = None
firstVertIndex = 12
numPolygonVerts = 12
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
