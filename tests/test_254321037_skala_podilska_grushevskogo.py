import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-0.7397915124893188, -0.868291974067688, 0.0)),
    Vector((-0.9375576972961426, -1.9369590282440186, 0.0)),
    Vector((-0.6738697290420532, -2.816383123397827, 0.0)),
    Vector((-0.19044145941734314, -3.406376361846924, 0.0)),
    Vector((0.7544412016868591, -3.873918294906616, 0.0)),
    Vector((1.7432719469070435, -3.851654052734375, 0.0)),
    Vector((2.6735050678253174, -3.3395841121673584, 0.0)),
    Vector((3.2155303955078125, -2.504687547683716, 0.0)),
    Vector((3.3034255504608154, -1.3469648361206055, 0.0)),
    Vector((2.907893180847168, -0.4786730408668518, 0.0)),
    Vector((2.307270050048828, 0.055660225450992584, 0.0)),
    Vector((1.4356346130371094, 0.35622256994247437, 0.0)),
    Vector((0.0, 0.0, 2.0)),
    Vector((-0.7397915124893188, -0.868291974067688, 2.0)),
    Vector((-0.9375576972961426, -1.9369590282440186, 2.0)),
    Vector((-0.6738697290420532, -2.816383123397827, 2.0)),
    Vector((-0.19044145941734314, -3.406376361846924, 2.0)),
    Vector((0.7544412016868591, -3.873918294906616, 2.0)),
    Vector((1.7432719469070435, -3.851654052734375, 2.0)),
    Vector((2.6735050678253174, -3.3395841121673584, 2.0)),
    Vector((3.2155303955078125, -2.504687547683716, 2.0)),
    Vector((3.3034255504608154, -1.3469648361206055, 2.0)),
    Vector((2.907893180847168, -0.4786730408668518, 2.0)),
    Vector((2.307270050048828, 0.055660225450992584, 2.0)),
    Vector((1.4356346130371094, 0.35622256994247437, 2.0)),
]
unitVectors = [
    Vector((-0.6485352516174316, -0.7611846923828125, 0.0)),
    Vector((-0.1819690465927124, -0.9833042025566101, 0.0)),
    Vector((0.28720876574516296, -0.9578680992126465, 0.0)),
    Vector((0.633792519569397, -0.7735031247138977, 0.0)),
    Vector((0.896278440952301, -0.4434918463230133, 0.0)),
    Vector((0.999746561050415, 0.022510020062327385, 0.0)),
    Vector((0.8760401606559753, 0.4822380840778351, 0.0)),
    Vector((0.5445239543914795, 0.8387452960014343, 0.0)),
    Vector((0.07570286840200424, 0.9971303939819336, 0.0)),
    Vector((-0.41454485058784485, 0.910028874874115, 0.0)),
    Vector((-0.7471337914466858, 0.6646737456321716, 0.0)),
    Vector((-0.9453734755516052, 0.32598912715911865, 0.0)),
    Vector((-0.9705682396888733, -0.24082611501216888, 0.0)),
]
holesInfo = None
firstVertIndex = 13
numPolygonVerts = 13
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
