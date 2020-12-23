import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((12.852293014526367, -7.224620342254639, 0.0)),
    Vector((14.946742057800293, -7.892531871795654, 0.0)),
    Vector((16.748260498046875, -7.859130859375, 0.0)),
    Vector((19.391948699951172, -7.280261039733887, 0.0)),
    Vector((20.74674415588379, -5.966686248779297, 0.0)),
    Vector((21.947750091552734, -4.263493061065674, 0.0)),
    Vector((22.482337951660156, -2.181816577911377, 0.0)),
    Vector((22.460359573364258, 0.033441122621297836, 0.0)),
    Vector((21.815906524658203, 2.1151130199432373, 0.0)),
    Vector((20.61488914489746, 3.9296162128448486, 0.0)),
    Vector((18.454532623291016, 5.28770637512207, 0.0)),
    Vector((16.792158126831055, 5.855430603027344, 0.0)),
    Vector((14.873473167419434, 5.822029113769531, 0.0)),
    Vector((0.534595787525177, 3.128077745437622, 0.0)),
    Vector((0.5345959663391113, 1.4805492162704468, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((12.852293014526367, -7.224620342254639, 18.0)),
    Vector((14.946742057800293, -7.892531871795654, 18.0)),
    Vector((16.748260498046875, -7.859130859375, 18.0)),
    Vector((19.391948699951172, -7.280261039733887, 18.0)),
    Vector((20.74674415588379, -5.966686248779297, 18.0)),
    Vector((21.947750091552734, -4.263493061065674, 18.0)),
    Vector((22.482337951660156, -2.181816577911377, 18.0)),
    Vector((22.460359573364258, 0.033441122621297836, 18.0)),
    Vector((21.815906524658203, 2.1151130199432373, 18.0)),
    Vector((20.61488914489746, 3.9296162128448486, 18.0)),
    Vector((18.454532623291016, 5.28770637512207, 18.0)),
    Vector((16.792158126831055, 5.855430603027344, 18.0)),
    Vector((14.873473167419434, 5.822029113769531, 18.0)),
    Vector((0.534595787525177, 3.128077745437622, 18.0)),
    Vector((0.5345959663391113, 1.4805492162704468, 18.0)),
    Vector((0.0, 0.0, 18.0))
]
unitVectors = [
    Vector((0.9527290463447571, -0.30382153391838074, 0.0)),
    Vector((0.9998281598091125, 0.018537292256951332, 0.0)),
    Vector((0.9768565893173218, 0.21389542520046234, 0.0)),
    Vector((0.7179443836212158, 0.6961004137992859, 0.0)),
    Vector((0.5762834548950195, 0.8172499537467957, 0.0)),
    Vector((0.24873536825180054, 0.9685714840888977, 0.0)),
    Vector((-0.0099208764731884, 0.9999508261680603, 0.0)),
    Vector((-0.29573652148246765, 0.9552695751190186, 0.0)),
    Vector((-0.5519446134567261, 0.8338807821273804, 0.0)),
    Vector((-0.8466097116470337, 0.5322141647338867, 0.0)),
    Vector((-0.9463351964950562, 0.3231867551803589, 0.0)),
    Vector((-0.9998484253883362, -0.01740589365363121, 0.0)),
    Vector((-0.9828049540519714, -0.1846468597650528, 0.0)),
    Vector((1.085346497120554e-07, -1.0, 0.0)),
    Vector((-0.33961817622184753, -0.940563440322876, 0.0)),
    Vector((0.8717144727706909, -0.4900142252445221, 0.0))
]
holesInfo = None
firstVertIndex = 16
numPolygonVerts = 16
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