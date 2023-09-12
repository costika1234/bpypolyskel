import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.19878263771533966, 0.17811119556427002, 0.0)),
    Vector((0.6511845588684082, -0.18924309313297272, 0.0)),
    Vector((1.17898690700531, -0.4898056089878082, 0.0)),
    Vector((1.7753349542617798, -0.6901805400848389, 0.0)),
    Vector((2.4128105640411377, -0.7903677821159363, 0.0)),
    Vector((3.063995122909546, -0.7681035399436951, 0.0)),
    Vector((3.6946160793304443, -0.6345197558403015, 0.0)),
    Vector((4.277254581451416, -0.38961637020111084, 0.0)),
    Vector((4.7844929695129395, -0.05565745010972023, 0.0)),
    Vector((5.2026214599609375, 0.3339611887931824, 0.0)),
    Vector((-0.3701465129852295, 6.567850112915039, 0.0)),
    Vector((-3.3930115699768066, 3.8516554832458496, 0.0)),
    Vector((-7.450909614562988, 8.4491548538208, 0.0)),
    Vector((-10.809651374816895, 5.476930618286133, 0.0)),
    Vector((-9.973396301269531, 4.530713081359863, 0.0)),
    Vector((-11.940661430358887, 2.794133424758911, 0.0)),
    Vector((-5.565919876098633, -4.942582130432129, 0.0)),
    Vector((0.19878263771533966, 0.17811119556427002, 6.066683769226074)),
    Vector((0.6511845588684082, -0.18924309313297272, 6.066683769226074)),
    Vector((1.17898690700531, -0.4898056089878082, 6.066683769226074)),
    Vector((1.7753349542617798, -0.6901805400848389, 6.066683769226074)),
    Vector((2.4128105640411377, -0.7903677821159363, 6.066683769226074)),
    Vector((3.063995122909546, -0.7681035399436951, 6.066683769226074)),
    Vector((3.6946160793304443, -0.6345197558403015, 6.066683769226074)),
    Vector((4.277254581451416, -0.38961637020111084, 6.066683769226074)),
    Vector((4.7844929695129395, -0.05565745010972023, 6.066683769226074)),
    Vector((5.2026214599609375, 0.3339611887931824, 6.066683769226074)),
    Vector((-0.3701465129852295, 6.567850112915039, 6.066683769226074)),
    Vector((-3.3930115699768066, 3.8516554832458496, 6.066683769226074)),
    Vector((-7.450909614562988, 8.4491548538208, 6.066683769226074)),
    Vector((-10.809651374816895, 5.476930618286133, 6.066683769226074)),
    Vector((-9.973396301269531, 4.530713081359863, 6.066683769226074)),
    Vector((-11.940661430358887, 2.794133424758911, 6.066683769226074)),
    Vector((-5.565919876098633, -4.942582130432129, 6.066683769226074)),
]
unitVectors = [
    Vector((0.7763006091117859, -0.6303628087043762, 0.0)),
    Vector((0.8689788579940796, -0.49484899640083313, 0.0)),
    Vector((0.9479212760925293, -0.3185047209262848, 0.0)),
    Vector((0.9878742098808289, -0.1552567481994629, 0.0)),
    Vector((0.9994159936904907, 0.03417040407657623, 0.0)),
    Vector((0.9782921075820923, 0.2072305977344513, 0.0)),
    Vector((0.9218717813491821, 0.3874950110912323, 0.0)),
    Vector((0.8352283835411072, 0.549903154373169, 0.0)),
    Vector((0.7316089272499084, 0.6817245483398438, 0.0)),
    Vector((-0.6664678454399109, 0.7455337643623352, 0.0)),
    Vector((-0.7438303232192993, -0.6683685183525085, 0.0)),
    Vector((-0.6617390513420105, 0.7497342228889465, 0.0)),
    Vector((-0.7488825917243958, -0.6627026200294495, 0.0)),
    Vector((0.6622257232666016, -0.7493044137954712, 0.0)),
    Vector((-0.7496946454048157, -0.6617839336395264, 0.0)),
    Vector((0.6359050869941711, -0.771767258644104, 0.0)),
    Vector((0.7476338148117065, 0.6641111969947815, 0.0)),
]
holesInfo = None
firstVertIndex = 17
numPolygonVerts = 17
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
