import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((1.9631397724151611, -13.336074829101562, 0.0)),
    Vector((3.9516100883483887, -13.024378776550293, 0.0)),
    Vector((5.154824256896973, -12.512308120727539, 0.0)),
    Vector((7.0672993659973145, -10.976096153259277, 0.0)),
    Vector((7.763895511627197, -10.007615089416504, 0.0)),
    Vector((8.270509719848633, -8.961211204528809, 0.0)),
    Vector((8.587142944335938, -7.859147548675537, 0.0)),
    Vector((8.72012710571289, -6.690292835235596, 0.0)),
    Vector((8.625134468078613, -5.421250820159912, 0.0)),
    Vector((8.31482982635498, -4.219000816345215, 0.0)),
    Vector((7.782881259918213, -3.083543062210083, 0.0)),
    Vector((7.0482869148254395, -2.0594048500061035, 0.0)),
    Vector((6.111047267913818, -1.1799824237823486, 0.0)),
    Vector((5.072484970092773, -0.5009347796440125, 0.0)),
    Vector((3.812279224395752, 1.6470766013298999e-06, 0.0)),
    Vector((2.6217334270477295, 0.23377171158790588, 0.0)),
    Vector((1.42485511302948, 0.25603505969047546, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((1.9631397724151611, -13.336074829101562, 16.0)),
    Vector((3.9516100883483887, -13.024378776550293, 16.0)),
    Vector((5.154824256896973, -12.512308120727539, 16.0)),
    Vector((7.0672993659973145, -10.976096153259277, 16.0)),
    Vector((7.763895511627197, -10.007615089416504, 16.0)),
    Vector((8.270509719848633, -8.961211204528809, 16.0)),
    Vector((8.587142944335938, -7.859147548675537, 16.0)),
    Vector((8.72012710571289, -6.690292835235596, 16.0)),
    Vector((8.625134468078613, -5.421250820159912, 16.0)),
    Vector((8.31482982635498, -4.219000816345215, 16.0)),
    Vector((7.782881259918213, -3.083543062210083, 16.0)),
    Vector((7.0482869148254395, -2.0594048500061035, 16.0)),
    Vector((6.111047267913818, -1.1799824237823486, 16.0)),
    Vector((5.072484970092773, -0.5009347796440125, 16.0)),
    Vector((3.812279224395752, 1.6470766013298999e-06, 16.0)),
    Vector((2.6217334270477295, 0.23377171158790588, 16.0)),
    Vector((1.42485511302948, 0.25603505969047546, 16.0)),
    Vector((0.0, 0.0, 16.0))
]
unitVectors = [
    Vector((0.9879363179206848, 0.15486067533493042, 0.0)),
    Vector((0.9201368093490601, 0.3915970027446747, 0.0)),
    Vector((0.7796279788017273, 0.6262428164482117, 0.0)),
    Vector((0.5839126110076904, 0.8118165731430054, 0.0)),
    Vector((0.435762882232666, 0.9000615477561951, 0.0)),
    Vector((0.27613818645477295, 0.9611179232597351, 0.0)),
    Vector((0.11304376274347305, 0.9935899972915649, 0.0)),
    Vector((-0.07464498281478882, 0.9972101449966431, 0.0)),
    Vector((-0.24991318583488464, 0.9682682156562805, 0.0)),
    Vector((-0.42423948645591736, 0.9055500030517578, 0.0)),
    Vector((-0.5828483700752258, 0.8125808238983154, 0.0)),
    Vector((-0.7292419672012329, 0.6842558979988098, 0.0)),
    Vector((-0.8369742035865784, 0.5472424030303955, 0.0)),
    Vector((-0.9292744398117065, 0.3693900406360626, 0.0)),
    Vector((-0.9812624454498291, 0.1926761418581009, 0.0)),
    Vector((-0.9998270273208618, 0.018597962334752083, 0.0)),
    Vector((-0.9842361807823181, -0.17685936391353607, 0.0)),
    Vector((0.14563575387001038, -0.9893382787704468, 0.0))
]
holesInfo = None
firstVertIndex = 18
numPolygonVerts = 18
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
