import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 3.0)),
    Vector((2.28084397315979, -0.9016873240470886, 3.0)),
    Vector((6.068546295166016, 3.4620399475097656, 3.0)),
    Vector((6.28772497177124, 4.664290428161621, 3.0)),
    Vector((6.02744722366333, 5.855408668518066, 3.0)),
    Vector((5.335659503936768, 6.846151351928711, 3.0)),
    Vector((4.315103054046631, 7.502935409545898, 3.0)),
    Vector((3.1164629459381104, 7.725573539733887, 3.0)),
    Vector((1.9246728420257568, 7.46953821182251, 3.0)),
    Vector((0.9246650338172913, 6.779356956481934, 3.0)),
    Vector((0.26712551712989807, 5.755217552185059, 3.0)),
    Vector((0.157535582780838, 5.142960548400879, 3.0)),
    Vector((0.0, 0.0, 9.0)),
    Vector((2.28084397315979, -0.9016873240470886, 9.0)),
    Vector((6.068546295166016, 3.4620399475097656, 9.0)),
    Vector((6.28772497177124, 4.664290428161621, 9.0)),
    Vector((6.02744722366333, 5.855408668518066, 9.0)),
    Vector((5.335659503936768, 6.846151351928711, 9.0)),
    Vector((4.315103054046631, 7.502935409545898, 9.0)),
    Vector((3.1164629459381104, 7.725573539733887, 9.0)),
    Vector((1.9246728420257568, 7.46953821182251, 9.0)),
    Vector((0.9246650338172913, 6.779356956481934, 9.0)),
    Vector((0.26712551712989807, 5.755217552185059, 9.0)),
    Vector((0.157535582780838, 5.142960548400879, 9.0))
]
unitVectors = [
    Vector((0.9299665689468384, -0.36764419078826904, 0.0)),
    Vector((0.6555041074752808, 0.7551916241645813, 0.0)),
    Vector((0.17935092747211456, 0.9837851524353027, 0.0)),
    Vector((-0.2134782075881958, 0.9769478440284729, 0.0)),
    Vector((-0.5724998712539673, 0.8199048042297363, 0.0)),
    Vector((-0.8409114480018616, 0.5411726236343384, 0.0)),
    Vector((-0.983183741569519, 0.18261878192424774, 0.0)),
    Vector((-0.977692723274231, -0.21004024147987366, 0.0)),
    Vector((-0.8230124711990356, -0.5680233240127563, 0.0)),
    Vector((-0.5402715802192688, -0.8414907455444336, 0.0)),
    Vector((-0.17619310319423676, -0.9843555688858032, 0.0)),
    Vector((-0.030616942793130875, -0.9995312094688416, 0.0))
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
