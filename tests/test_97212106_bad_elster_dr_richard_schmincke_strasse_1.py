import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((4.2541351318359375, -0.35622066259384155, 0.0)),
    Vector((4.062060832977295, -2.582610607147217, 0.0)),
    Vector((7.4127278327941895, -2.872037649154663, 0.0)),
    Vector((7.597687244415283, -0.7124392986297607, 0.0)),
    Vector((11.787797927856445, -1.06865394115448, 0.0)),
    Vector((11.88739013671875, 0.10020087659358978, 0.0)),
    Vector((13.295950889587402, -0.02224721759557724, 0.0)),
    Vector((13.900616645812988, 7.057673931121826, 0.0)),
    Vector((12.484944343566895, 7.180121898651123, 0.0)),
    Vector((13.46663761138916, 18.723955154418945, 0.0)),
    Vector((2.689058780670166, 19.63675880432129, 0.0)),
    Vector((2.3902759552001953, 16.096799850463867, 0.0)),
    Vector((1.3516441583633423, 16.185853958129883, 0.0)),
    Vector((0.9532656669616699, 11.499303817749023, 0.0)),
    Vector((0.5477720499038696, 11.532699584960938, 0.0)),
    Vector((-0.014227863401174545, 4.88692569732666, 0.0)),
    Vector((0.4126080274581909, 4.853529930114746, 0.0)),
    Vector((0.0, 0.0, 6.097714424133301)),
    Vector((4.2541351318359375, -0.35622066259384155, 6.097714424133301)),
    Vector((4.062060832977295, -2.582610607147217, 6.097714424133301)),
    Vector((7.4127278327941895, -2.872037649154663, 6.097714424133301)),
    Vector((7.597687244415283, -0.7124392986297607, 6.097714424133301)),
    Vector((11.787797927856445, -1.06865394115448, 6.097714424133301)),
    Vector((11.88739013671875, 0.10020087659358978, 6.097714424133301)),
    Vector((13.295950889587402, -0.02224721759557724, 6.097714424133301)),
    Vector((13.900616645812988, 7.057673931121826, 6.097714424133301)),
    Vector((12.484944343566895, 7.180121898651123, 6.097714424133301)),
    Vector((13.46663761138916, 18.723955154418945, 6.097714424133301)),
    Vector((2.689058780670166, 19.63675880432129, 6.097714424133301)),
    Vector((2.3902759552001953, 16.096799850463867, 6.097714424133301)),
    Vector((1.3516441583633423, 16.185853958129883, 6.097714424133301)),
    Vector((0.9532656669616699, 11.499303817749023, 6.097714424133301)),
    Vector((0.5477720499038696, 11.532699584960938, 6.097714424133301)),
    Vector((-0.014227863401174545, 4.88692569732666, 6.097714424133301)),
    Vector((0.4126080274581909, 4.853529930114746, 6.097714424133301))
]
unitVectors = [
    Vector((0.9965124726295471, -0.08344312757253647, 0.0)),
    Vector((-0.08595236390829086, -0.9962992072105408, 0.0)),
    Vector((0.9962901473045349, -0.08605848252773285, 0.0)),
    Vector((0.08533289283514023, 0.99635249376297, 0.0)),
    Vector((0.996405839920044, -0.08470763266086578, 0.0)),
    Vector((0.08489731699228287, 0.996389627456665, 0.0)),
    Vector((0.9962428212165833, -0.08660473674535751, 0.0)),
    Vector((0.08509593456983566, 0.9963727593421936, 0.0)),
    Vector((-0.9962802529335022, 0.08617283403873444, 0.0)),
    Vector((0.08473465591669083, 0.9964035749435425, 0.0)),
    Vector((-0.9964325428009033, 0.08439254760742188, 0.0)),
    Vector((-0.08410386741161346, -0.996457040309906, 0.0)),
    Vector((-0.9963444471359253, 0.08542831242084503, 0.0)),
    Vector((-0.08469917625188828, -0.9964065551757812, 0.0)),
    Vector((-0.9966257810592651, 0.0820804089307785, 0.0)),
    Vector((-0.08426424860954285, -0.9964434504508972, 0.0)),
    Vector((0.9969531893730164, -0.07800191640853882, 0.0)),
    Vector((-0.08470641076564789, -0.9964059591293335, 0.0))
]
holesInfo = None
firstVertIndex = 18
numPolygonVerts = 18
faces = []


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