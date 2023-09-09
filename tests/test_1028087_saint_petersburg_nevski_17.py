import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-31.654027938842773, -30.77970314025879, 0.0)),
    Vector((35.56366729736328, -60.9027214050293, 0.0)),
    Vector((40.78905487060547, -48.5796012878418, 0.0)),
    Vector((52.78971862792969, 1.146968126296997, 0.0)),
    Vector((-18.609600067138672, 18.100595474243164, 0.0)),
    Vector((-31.654027938842773, -30.77970314025879, 9.376729011535645)),
    Vector((35.56366729736328, -60.9027214050293, 9.376729011535645)),
    Vector((40.78905487060547, -48.5796012878418, 9.376729011535645)),
    Vector((52.78971862792969, 1.146968126296997, 9.376729011535645)),
    Vector((-18.609600067138672, 18.100595474243164, 9.376729011535645)),
    Vector((0.0, 0.0, 0.0)),
    Vector((41.0786247253418, -10.09644889831543, 0.0)),
    Vector((38.123043060302734, -21.506729125976562, 0.0)),
    Vector((36.48345947265625, -21.094863891601562, 0.0)),
    Vector((33.6728515625, -32.315895080566406, 0.0)),
    Vector((35.184173583984375, -32.77228927612305, 0.0)),
    Vector((32.70257568359375, -42.55729675292969, 0.0)),
    Vector((25.10690689086914, -40.45341873168945, 0.0)),
    Vector((25.575349807739258, -38.661170959472656, 0.0)),
    Vector((-3.808966875076294, -29.9449405670166, 0.0)),
    Vector((-4.472610950469971, -31.982086181640625, 0.0)),
    Vector((-12.575724601745605, -29.63322639465332, 0.0)),
    Vector((-6.112153053283691, -5.799740314483643, 0.0)),
    Vector((-1.712072491645813, -6.901808261871338, 0.0)),
    Vector((0.0, 0.0, 9.376729011535645)),
    Vector((41.0786247253418, -10.09644889831543, 9.376729011535645)),
    Vector((38.123043060302734, -21.506729125976562, 9.376729011535645)),
    Vector((36.48345947265625, -21.094863891601562, 9.376729011535645)),
    Vector((33.6728515625, -32.315895080566406, 9.376729011535645)),
    Vector((35.184173583984375, -32.77228927612305, 9.376729011535645)),
    Vector((32.70257568359375, -42.55729675292969, 9.376729011535645)),
    Vector((25.10690689086914, -40.45341873168945, 9.376729011535645)),
    Vector((25.575349807739258, -38.661170959472656, 9.376729011535645)),
    Vector((-3.808966875076294, -29.9449405670166, 9.376729011535645)),
    Vector((-4.472610950469971, -31.982086181640625, 9.376729011535645)),
    Vector((-12.575724601745605, -29.63322639465332, 9.376729011535645)),
    Vector((-6.112153053283691, -5.799740314483643, 9.376729011535645)),
    Vector((-1.712072491645813, -6.901808261871338, 9.376729011535645))
]
unitVectors = [
    Vector((0.9125551581382751, -0.40895354747772217, 0.0)),
    Vector((0.39038506150245667, 0.9206517338752747, 0.0)),
    Vector((0.23459801077842712, 0.9720924496650696, 0.0)),
    Vector((-0.972947895526886, 0.23102453351020813, 0.0)),
    Vector((-0.2578412890434265, -0.9661872386932373, 0.0)),
    Vector((0.9710983037948608, -0.2386799454689026, 0.0)),
    Vector((-0.25075235962867737, -0.9680513143539429, 0.0)),
    Vector((-0.9698678255081177, 0.24363188445568085, 0.0)),
    Vector((-0.24297089874744415, -0.970033586025238, 0.0)),
    Vector((0.9573020935058594, -0.2890893816947937, 0.0)),
    Vector((-0.24582967162132263, -0.9693130254745483, 0.0)),
    Vector((-0.9637149572372437, 0.26693353056907654, 0.0)),
    Vector((0.25287675857543945, 0.9674984812736511, 0.0)),
    Vector((-0.9587113261222839, 0.284381240606308, 0.0)),
    Vector((-0.3097495138645172, -0.9508182406425476, 0.0)),
    Vector((-0.960462212562561, 0.278410404920578, 0.0)),
    Vector((0.261742502450943, 0.9651376605033875, 0.0)),
    Vector((0.9700362086296082, -0.24296049773693085, 0.0)),
    Vector((0.24076437950134277, 0.9705836772918701, 0.0))
]
holesInfo = [
    (24, 14)
]
firstVertIndex = 5
numPolygonVerts = 5
faces = []

bpypolyskel.debugOutputs["skeleton"] = 1


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
    assert not bpypolyskel.check_edge_crossing(bpypolyskel.debugOutputs["skeleton"])
