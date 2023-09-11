import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-4.8646931648254395, -12.512308120727539, 0.0)),
    Vector((0.6280999183654785, -14.64964485168457, 0.0)),
    Vector((2.8202919960021973, -14.939074516296387, 0.0)),
    Vector((4.975536346435547, -14.449267387390137, 0.0)),
    Vector((6.85983419418335, -13.30267333984375, 0.0)),
    Vector((8.276134490966797, -11.599482536315918, 0.0)),
    Vector((9.070490837097168, -9.540070533752441, 0.0)),
    Vector((9.162853240966797, -7.324812412261963, 0.0)),
    Vector((8.540907859802246, -5.209743499755859, 0.0)),
    Vector((7.272392272949219, -3.395238161087036, 0.0)),
    Vector((5.49277925491333, -2.1373307704925537, 0.0)),
    Vector((0.0, 0.0, 3.123523235321045)),
    Vector((-4.8646931648254395, -12.512308120727539, 3.123523235321045)),
    Vector((0.6280999183654785, -14.64964485168457, 3.123523235321045)),
    Vector((2.8202919960021973, -14.939074516296387, 3.123523235321045)),
    Vector((4.975536346435547, -14.449267387390137, 3.123523235321045)),
    Vector((6.85983419418335, -13.30267333984375, 3.123523235321045)),
    Vector((8.276134490966797, -11.599482536315918, 3.123523235321045)),
    Vector((9.070490837097168, -9.540070533752441, 3.123523235321045)),
    Vector((9.162853240966797, -7.324812412261963, 3.123523235321045)),
    Vector((8.540907859802246, -5.209743499755859, 3.123523235321045)),
    Vector((7.272392272949219, -3.395238161087036, 3.123523235321045)),
    Vector((5.49277925491333, -2.1373307704925537, 3.123523235321045))
]
unitVectors = [
    Vector((-0.36236831545829773, -0.932034969329834, 0.0)),
    Vector((0.9319329857826233, -0.36263054609298706, 0.0)),
    Vector((0.9913966655731201, -0.13089163601398468, 0.0)),
    Vector((0.9751348495483398, 0.22161199152469635, 0.0)),
    Vector((0.8542730212211609, 0.5198245644569397, 0.0)),
    Vector((0.639378011226654, 0.7688925862312317, 0.0)),
    Vector((0.35987669229507446, 0.9329999089241028, 0.0)),
    Vector((0.041657548397779465, 0.9991318583488464, 0.0)),
    Vector((-0.2821105122566223, 0.9593819975852966, 0.0)),
    Vector((-0.5729656219482422, 0.8195793628692627, 0.0)),
    Vector((-0.8165979385375977, 0.5772067308425903, 0.0)),
    Vector((-0.9319329857826233, 0.3626304566860199, 0.0))
]
holesInfo = None
firstVertIndex = 12
numPolygonVerts = 12
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
