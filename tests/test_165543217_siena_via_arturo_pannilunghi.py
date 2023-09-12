import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-1.7493665218353271, 0.8015005588531494, 0.0)),
    Vector((-15.849656105041504, -30.033979415893555, 0.0)),
    Vector((-14.003096580505371, -30.88001251220703, 0.0)),
    Vector((-14.837292671203613, -32.69451904296875, 0.0)),
    Vector((-4.494925498962402, -37.41447830200195, 0.0)),
    Vector((-3.676928997039795, -35.61110305786133, 0.0)),
    Vector((-2.219115972518921, -36.279022216796875, 0.0)),
    Vector((11.881124496459961, -5.443512439727783, 0.0)),
    Vector((10.326129913330078, -4.731070518493652, 0.0)),
    Vector((11.217008590698242, -2.782978057861328, 0.0)),
    Vector((0.882781982421875, 1.936959147453308, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((-1.7493665218353271, 0.8015005588531494, 18.149648666381836)),
    Vector((-15.849656105041504, -30.033979415893555, 18.149648666381836)),
    Vector((-14.003096580505371, -30.88001251220703, 18.149648666381836)),
    Vector((-14.837292671203613, -32.69451904296875, 18.149648666381836)),
    Vector((-4.494925498962402, -37.41447830200195, 18.149648666381836)),
    Vector((-3.676928997039795, -35.61110305786133, 18.149648666381836)),
    Vector((-2.219115972518921, -36.279022216796875, 18.149648666381836)),
    Vector((11.881124496459961, -5.443512439727783, 18.149648666381836)),
    Vector((10.326129913330078, -4.731070518493652, 18.149648666381836)),
    Vector((11.217008590698242, -2.782978057861328, 18.149648666381836)),
    Vector((0.882781982421875, 1.936959147453308, 18.149648666381836)),
    Vector((0.0, 0.0, 18.149648666381836)),
]
unitVectors = [
    Vector((-0.41585904359817505, -0.9094290733337402, 0.0)),
    Vector((0.9091219902038574, -0.41652989387512207, 0.0)),
    Vector((-0.41770848631858826, -0.9085810780525208, 0.0)),
    Vector((0.9097397327423096, -0.4151790738105774, 0.0)),
    Vector((0.41308316588401794, 0.9106932878494263, 0.0)),
    Vector((0.9091227054595947, -0.4165283739566803, 0.0)),
    Vector((0.4158575236797333, 0.9094298481941223, 0.0)),
    Vector((-0.9091233015060425, 0.4165272116661072, 0.0)),
    Vector((0.4158841669559479, 0.9094176292419434, 0.0)),
    Vector((-0.9096168279647827, 0.41544806957244873, 0.0)),
    Vector((-0.41471609473228455, -0.9099507927894592, 0.0)),
    Vector((-0.9091224074363708, 0.41652911901474, 0.0)),
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
