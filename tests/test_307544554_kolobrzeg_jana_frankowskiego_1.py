import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((4.417402744293213, 2.460162878036499, 0.0)),
    Vector((-0.8274468779563904, 11.866658210754395, 0.0)),
    Vector((-5.244843006134033, 9.406499862670898, 0.0)),
    Vector((-6.573972702026367, 8.438021659851074, 0.0)),
    Vector((-7.590366840362549, 7.046530246734619, 0.0)),
    Vector((-8.105081558227539, 5.410134315490723, 0.0)),
    Vector((-8.085538864135742, 3.6846821308135986, 0.0)),
    Vector((-7.518706321716309, 2.0594167709350586, 0.0)),
    Vector((-6.469738006591797, 0.6901853680610657, 0.0)),
    Vector((-5.0428786277771, -0.267164021730423, 0.0)),
    Vector((-3.3879806995391846, -0.7347074151039124, 0.0)),
    Vector((-1.66792893409729, -0.65678471326828, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((4.417402744293213, 2.460162878036499, 5.750707149505615)),
    Vector((-0.8274468779563904, 11.866658210754395, 5.750707149505615)),
    Vector((-5.244843006134033, 9.406499862670898, 5.750707149505615)),
    Vector((-6.573972702026367, 8.438021659851074, 5.750707149505615)),
    Vector((-7.590366840362549, 7.046530246734619, 5.750707149505615)),
    Vector((-8.105081558227539, 5.410134315490723, 5.750707149505615)),
    Vector((-8.085538864135742, 3.6846821308135986, 5.750707149505615)),
    Vector((-7.518706321716309, 2.0594167709350586, 5.750707149505615)),
    Vector((-6.469738006591797, 0.6901853680610657, 5.750707149505615)),
    Vector((-5.0428786277771, -0.267164021730423, 5.750707149505615)),
    Vector((-3.3879806995391846, -0.7347074151039124, 5.750707149505615)),
    Vector((-1.66792893409729, -0.65678471326828, 5.750707149505615)),
    Vector((0.0, 0.0, 5.750707149505615))
]
unitVectors = [
    Vector((-0.486991822719574, 0.8734065890312195, 0.0)),
    Vector((-0.8736488819122314, -0.48655691742897034, 0.0)),
    Vector((-0.8082039952278137, -0.5889025926589966, 0.0)),
    Vector((-0.5898406505584717, -0.8075196146965027, 0.0)),
    Vector((-0.300048828125, -0.9539239406585693, 0.0)),
    Vector((0.011325403116643429, -0.9999358654022217, 0.0)),
    Vector((0.32930976152420044, -0.9442219734191895, 0.0)),
    Vector((0.6081482172012329, -0.7938234210014343, 0.0)),
    Vector((0.8304054737091064, -0.5571594834327698, 0.0)),
    Vector((0.9623315334320068, -0.27187883853912354, 0.0)),
    Vector((0.998975396156311, 0.0452561154961586, 0.0)),
    Vector((0.9304613471031189, 0.3663901686668396, 0.0)),
    Vector((0.8736488223075867, 0.4865570366382599, 0.0))
]
holesInfo = None
firstVertIndex = 13
numPolygonVerts = 13
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