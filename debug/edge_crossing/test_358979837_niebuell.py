import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-2.7348222732543945, -4.909188747406006, 0.0)),
    Vector((-3.9930965900421143, -4.207874774932861, 0.0)),
    Vector((-7.793612003326416, -11.031754493713379, 0.0)),
    Vector((14.09787368774414, -23.23235511779785, 0.0)),
    Vector((17.898365020751953, -16.408456802368164, 0.0)),
    Vector((15.381804466247559, -14.994709014892578, 0.0)),
    Vector((21.300796508789062, -4.374805450439453, 0.0)),
    Vector((14.354592323303223, -0.5009148120880127, 0.0)),
    Vector((8.429170608520508, -11.131940841674805, 0.0)),
    Vector((4.217792987823486, -8.783105850219727, 0.0)),
    Vector((6.952609539031982, -3.873912811279297, 0.0)),
    Vector((0.0, 0.0, 2.8213601112365723)),
    Vector((-2.7348222732543945, -4.909188747406006, 2.8213601112365723)),
    Vector((-3.9930965900421143, -4.207874774932861, 2.8213601112365723)),
    Vector((-7.793612003326416, -11.031754493713379, 2.8213601112365723)),
    Vector((14.09787368774414, -23.23235511779785, 2.8213601112365723)),
    Vector((17.898365020751953, -16.408456802368164, 2.8213601112365723)),
    Vector((15.381804466247559, -14.994709014892578, 2.8213601112365723)),
    Vector((21.300796508789062, -4.374805450439453, 2.8213601112365723)),
    Vector((14.354592323303223, -0.5009148120880127, 2.8213601112365723)),
    Vector((8.429170608520508, -11.131940841674805, 2.8213601112365723)),
    Vector((4.217792987823486, -8.783105850219727, 2.8213601112365723)),
    Vector((6.952609539031982, -3.873912811279297, 2.8213601112365723))
]
unitVectors = [
    Vector((-0.486661821603775, -0.8735904693603516, 0.0)),
    Vector((-0.8734866380691528, 0.48684805631637573, 0.0)),
    Vector((-0.48656922578811646, -0.8736420273780823, 0.0)),
    Vector((0.8735015392303467, -0.48682138323783875, 0.0)),
    Vector((0.48656585812568665, 0.8736438751220703, 0.0)),
    Vector((-0.8718444108963013, 0.48978281021118164, 0.0)),
    Vector((0.48683950304985046, 0.8734914064407349, 0.0)),
    Vector((-0.8733614087104797, 0.48707273602485657, 0.0)),
    Vector((-0.4868539571762085, -0.8734833598136902, 0.0)),
    Vector((-0.873347818851471, 0.4870971143245697, 0.0)),
    Vector((0.4866607189178467, 0.8735911250114441, 0.0)),
    Vector((-0.873551070690155, 0.48673245310783386, 0.0))
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
