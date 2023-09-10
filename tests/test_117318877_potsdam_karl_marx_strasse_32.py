import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((8.169790267944336, 11.655157089233398, 0.0)),
    Vector((-2.906620740890503, 19.414119720458984, 0.0)),
    Vector((-2.7300500869750977, 19.670154571533203, 0.0)),
    Vector((-7.1918439865112305, 22.798236846923828, 0.0)),
    Vector((-8.176565170288086, 21.395612716674805, 0.0)),
    Vector((-8.332761764526367, 21.506933212280273, 0.0)),
    Vector((-12.15620231628418, 16.05228614807129, 0.0)),
    Vector((-12.64516544342041, 16.397377014160156, 0.0)),
    Vector((-15.191866874694824, 12.757237434387207, 0.0)),
    Vector((-14.668947219848633, 12.389881134033203, 0.0)),
    Vector((-15.667255401611328, 10.976126670837402, 0.0)),
    Vector((0.0, 0.0, 5.866687297821045)),
    Vector((8.169790267944336, 11.655157089233398, 5.866687297821045)),
    Vector((-2.906620740890503, 19.414119720458984, 5.866687297821045)),
    Vector((-2.7300500869750977, 19.670154571533203, 5.866687297821045)),
    Vector((-7.1918439865112305, 22.798236846923828, 5.866687297821045)),
    Vector((-8.176565170288086, 21.395612716674805, 5.866687297821045)),
    Vector((-8.332761764526367, 21.506933212280273, 5.866687297821045)),
    Vector((-12.15620231628418, 16.05228614807129, 5.866687297821045)),
    Vector((-12.64516544342041, 16.397377014160156, 5.866687297821045)),
    Vector((-15.191866874694824, 12.757237434387207, 5.866687297821045)),
    Vector((-14.668947219848633, 12.389881134033203, 5.866687297821045)),
    Vector((-15.667255401611328, 10.976126670837402, 5.866687297821045))
]
unitVectors = [
    Vector((0.5739893913269043, 0.8188627362251282, 0.0)),
    Vector((-0.8190416693687439, 0.5737339854240417, 0.0)),
    Vector((0.5677218437194824, 0.823220431804657, 0.0)),
    Vector((-0.8188155889511108, 0.5740566849708557, 0.0)),
    Vector((-0.574591338634491, -0.8184404969215393, 0.0)),
    Vector((-0.8143457770347595, 0.5803799629211426, 0.0)),
    Vector((-0.5739848613739014, -0.8188658952713013, 0.0)),
    Vector((-0.8170146942138672, 0.5766167044639587, 0.0)),
    Vector((-0.5732514262199402, -0.8193795680999756, 0.0)),
    Vector((0.8182658553123474, -0.5748400092124939, 0.0)),
    Vector((-0.5768235325813293, -0.816868782043457, 0.0)),
    Vector((0.8190096616744995, -0.5737797617912292, 0.0))
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
