import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 7.081154551613622e-10, 0.0)),
    Vector((0.5643750429153442, -0.8460280895233154, 0.0)),
    Vector((1.3262815475463867, -1.6252644062042236, 0.0)),
    Vector((2.05291485786438, -2.137333869934082, 0.0)),
    Vector((2.814821481704712, -2.482423782348633, 0.0)),
    Vector((3.6049468517303467, -2.616006851196289, 0.0)),
    Vector((4.5432209968566895, -2.6160061359405518, 0.0)),
    Vector((5.516767978668213, -2.437893867492676, 0.0)),
    Vector((6.469150543212891, -2.037142753601074, 0.0)),
    Vector((7.2804388999938965, -1.3914885520935059, 0.0)),
    Vector((7.894196033477783, -0.7792304754257202, 0.0)),
    Vector((10.017655372619629, 0.02227349951863289, 0.0)),
    Vector((5.601410388946533, 10.39724349975586, 0.0)),
    Vector((-5.77072811126709, 5.5771098136901855, 0.0)),
    Vector((-3.887131690979004, 1.1354602575302124, 0.0)),
    Vector((-0.3738982379436493, 2.6048760414123535, 0.0)),
    Vector((-0.3950623571872711, 1.7365840673446655, 0.0)),
    Vector((-0.28924211859703064, 0.9128198027610779, 0.0)),
    Vector((0.0, 7.081154551613622e-10, 5.956595420837402)),
    Vector((0.5643750429153442, -0.8460280895233154, 5.956595420837402)),
    Vector((1.3262815475463867, -1.6252644062042236, 5.956595420837402)),
    Vector((2.05291485786438, -2.137333869934082, 5.956595420837402)),
    Vector((2.814821481704712, -2.482423782348633, 5.956595420837402)),
    Vector((3.6049468517303467, -2.616006851196289, 5.956595420837402)),
    Vector((4.5432209968566895, -2.6160061359405518, 5.956595420837402)),
    Vector((5.516767978668213, -2.437893867492676, 5.956595420837402)),
    Vector((6.469150543212891, -2.037142753601074, 5.956595420837402)),
    Vector((7.2804388999938965, -1.3914885520935059, 5.956595420837402)),
    Vector((7.894196033477783, -0.7792304754257202, 5.956595420837402)),
    Vector((10.017655372619629, 0.02227349951863289, 5.956595420837402)),
    Vector((5.601410388946533, 10.39724349975586, 5.956595420837402)),
    Vector((-5.77072811126709, 5.5771098136901855, 5.956595420837402)),
    Vector((-3.887131690979004, 1.1354602575302124, 5.956595420837402)),
    Vector((-0.3738982379436493, 2.6048760414123535, 5.956595420837402)),
    Vector((-0.3950623571872711, 1.7365840673446655, 5.956595420837402)),
    Vector((-0.28924211859703064, 0.9128198027610779, 5.956595420837402))
]
unitVectors = [
    Vector((0.554942786693573, -0.8318886160850525, 0.0)),
    Vector((0.69911128282547, -0.7150128483772278, 0.0)),
    Vector((0.8174171447753906, -0.5760462284088135, 0.0)),
    Vector((0.910920262336731, -0.41258254647254944, 0.0)),
    Vector((0.9860076308250427, -0.16670003533363342, 0.0)),
    Vector((1.0, 7.623099804732192e-07, 0.0)),
    Vector((0.983673095703125, 0.1799648553133011, 0.0)),
    Vector((0.9217225909233093, 0.38784974813461304, 0.0)),
    Vector((0.7824548482894897, 0.6227073669433594, 0.0)),
    Vector((0.7079708576202393, 0.7062417268753052, 0.0)),
    Vector((0.9355728030204773, 0.35313379764556885, 0.0)),
    Vector((-0.3916575610637665, 0.9201110601425171, 0.0)),
    Vector((-0.9207101464271545, -0.39024725556373596, 0.0)),
    Vector((0.3904199004173279, -0.9206368923187256, 0.0)),
    Vector((0.9225568771362305, 0.38586097955703735, 0.0)),
    Vector((-0.024367190897464752, -0.9997031092643738, 0.0)),
    Vector((0.12741239368915558, -0.9918497800827026, 0.0)),
    Vector((0.3020649552345276, -0.9532873630523682, 0.0))
]
holesInfo = None
firstVertIndex = 18
numPolygonVerts = 18
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
