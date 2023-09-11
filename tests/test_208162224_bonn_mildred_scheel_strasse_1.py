import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-23.546546936035156, -26.493986129760742, 0.0)),
    Vector((-56.103050231933594, -47.688968658447266, 0.0)),
    Vector((-49.8851318359375, -56.98420715332031, 0.0)),
    Vector((-16.58835792541504, -35.121273040771484, 0.0)),
    Vector((8.269460678100586, -7.191232681274414, 0.0)),
    Vector((8.953292846679688, -5.933320999145508, 0.0)),
    Vector((9.263483047485352, -4.541827201843262, 0.0)),
    Vector((9.263481140136719, -3.1725971698760986, 0.0)),
    Vector((8.932135581970215, -1.5139374732971191, 0.0)),
    Vector((8.07910442352295, -0.18923687934875488, 0.0)),
    Vector((6.8524322509765625, 0.8905604481697083, 0.0)),
    Vector((5.054725646972656, 1.6030031442642212, 0.0)),
    Vector((3.2147209644317627, 1.7031892538070679, 0.0)),
    Vector((1.4170153141021729, 1.1354589462280273, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((-23.546546936035156, -26.493986129760742, 14.095224380493164)),
    Vector((-56.103050231933594, -47.688968658447266, 14.095224380493164)),
    Vector((-49.8851318359375, -56.98420715332031, 14.095224380493164)),
    Vector((-16.58835792541504, -35.121273040771484, 14.095224380493164)),
    Vector((8.269460678100586, -7.191232681274414, 14.095224380493164)),
    Vector((8.953292846679688, -5.933320999145508, 14.095224380493164)),
    Vector((9.263483047485352, -4.541827201843262, 14.095224380493164)),
    Vector((9.263481140136719, -3.1725971698760986, 14.095224380493164)),
    Vector((8.932135581970215, -1.5139374732971191, 14.095224380493164)),
    Vector((8.07910442352295, -0.18923687934875488, 14.095224380493164)),
    Vector((6.8524322509765625, 0.8905604481697083, 14.095224380493164)),
    Vector((5.054725646972656, 1.6030031442642212, 14.095224380493164)),
    Vector((3.2147209644317627, 1.7031892538070679, 14.095224380493164)),
    Vector((1.4170153141021729, 1.1354589462280273, 14.095224380493164)),
    Vector((0.0, 0.0, 14.095224380493164))
]
unitVectors = [
    Vector((-0.8380522131919861, -0.5455899834632874, 0.0)),
    Vector((0.5560052990913391, -0.8311787843704224, 0.0)),
    Vector((0.8359105587005615, 0.5488656759262085, 0.0)),
    Vector((0.6648285984992981, 0.7469959259033203, 0.0)),
    Vector((0.47761282324790955, 0.8785704374313354, 0.0)),
    Vector((0.21757836639881134, 0.9760429263114929, 0.0)),
    Vector((-1.3930082332080929e-06, 1.0, 0.0)),
    Vector((-0.19589649140834808, 0.9806246161460876, 0.0)),
    Vector((-0.5414032340049744, 0.8407631516456604, 0.0)),
    Vector((-0.7506146430969238, 0.6607401967048645, 0.0)),
    Vector((-0.9296559691429138, 0.36842864751815796, 0.0)),
    Vector((-0.9985209703445435, 0.05436830222606659, 0.0)),
    Vector((-0.9535775184631348, -0.3011476695537567, 0.0)),
    Vector((-0.7803725004196167, -0.6253150105476379, 0.0)),
    Vector((-0.664306104183197, -0.7474606037139893, 0.0))
]
holesInfo = None
firstVertIndex = 15
numPolygonVerts = 15
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
