import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.5733487606048584, 1.9592230319976807, 0.0)),
    Vector((0.013030651025474072, 2.9833624362945557, 0.0)),
    Vector((-1.935051441192627, 3.4731686115264893, 0.0)),
    Vector((-6.352445125579834, 1.0018798112869263, 0.0)),
    Vector((-8.619782447814941, -0.8237561583518982, 0.0)),
    Vector((-10.261651992797852, -3.2505176067352295, 0.0)),
    Vector((-11.069560050964355, -6.055767059326172, 0.0)),
    Vector((-10.971837043762207, -8.98346996307373, 0.0)),
    Vector((-9.968480110168457, -11.733063697814941, 0.0)),
    Vector((-8.170248031616211, -14.04851245880127, 0.0)),
    Vector((-5.740025043487549, -15.67378044128418, 0.0)),
    Vector((-2.925393581390381, -16.48641586303711, 0.0)),
    Vector((0.0, -16.375097274780273, 0.0)),
    Vector((2.755993604660034, -15.328693389892578, 0.0)),
    Vector((7.179913520812988, -12.86852741241455, 0.0)),
    Vector((7.662045955657959, -11.020623207092285, 0.0)),
    Vector((7.0626325607299805, -9.940825462341309, 0.0)),
    Vector((5.244851112365723, -9.406494140625, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((0.5733487606048584, 1.9592230319976807, 17.482189178466797)),
    Vector((0.013030651025474072, 2.9833624362945557, 17.482189178466797)),
    Vector((-1.935051441192627, 3.4731686115264893, 17.482189178466797)),
    Vector((-6.352445125579834, 1.0018798112869263, 17.482189178466797)),
    Vector((-8.619782447814941, -0.8237561583518982, 17.482189178466797)),
    Vector((-10.261651992797852, -3.2505176067352295, 17.482189178466797)),
    Vector((-11.069560050964355, -6.055767059326172, 17.482189178466797)),
    Vector((-10.971837043762207, -8.98346996307373, 17.482189178466797)),
    Vector((-9.968480110168457, -11.733063697814941, 17.482189178466797)),
    Vector((-8.170248031616211, -14.04851245880127, 17.482189178466797)),
    Vector((-5.740025043487549, -15.67378044128418, 17.482189178466797)),
    Vector((-2.925393581390381, -16.48641586303711, 17.482189178466797)),
    Vector((0.0, -16.375097274780273, 17.482189178466797)),
    Vector((2.755993604660034, -15.328693389892578, 17.482189178466797)),
    Vector((7.179913520812988, -12.86852741241455, 17.482189178466797)),
    Vector((7.662045955657959, -11.020623207092285, 17.482189178466797)),
    Vector((7.0626325607299805, -9.940825462341309, 17.482189178466797)),
    Vector((5.244851112365723, -9.406494140625, 17.482189178466797)),
    Vector((0.0, 0.0, 17.482189178466797))
]
unitVectors = [
    Vector((-0.47997185587882996, 0.8772839903831482, 0.0)),
    Vector((-0.9698153138160706, 0.24384061992168427, 0.0)),
    Vector((-0.8727123737335205, -0.4882345497608185, 0.0)),
    Vector((-0.7788931727409363, -0.6271564960479736, 0.0)),
    Vector((-0.560364842414856, -0.8282459378242493, 0.0)),
    Vector((-0.27674996852874756, -0.9609419107437134, 0.0)),
    Vector((0.03336014971137047, -0.9994433522224426, 0.0)),
    Vector((0.34280040860176086, -0.9394083023071289, 0.0)),
    Vector((0.6133724451065063, -0.7897937893867493, 0.0)),
    Vector((0.831241250038147, -0.5559118390083313, 0.0)),
    Vector((0.9607579112052917, -0.277388334274292, 0.0)),
    Vector((0.9992767572402954, 0.03802499547600746, 0.0)),
    Vector((0.9348819851875305, 0.35495877265930176, 0.0)),
    Vector((0.873953104019165, 0.4860101044178009, 0.0)),
    Vector((0.2524564564228058, 0.9676083326339722, 0.0)),
    Vector((-0.48534950613975525, 0.8743202686309814, 0.0)),
    Vector((-0.9594098329544067, 0.2820155918598175, 0.0)),
    Vector((-0.4869919717311859, 0.8734065294265747, 0.0)),
    Vector((0.28086158633232117, 0.9597483277320862, 0.0))
]
holesInfo = None
firstVertIndex = 19
numPolygonVerts = 19
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
