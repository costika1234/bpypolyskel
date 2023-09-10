import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.4302951693534851, -4.575231075286865, 0.0)),
    Vector((13.30997371673584, -3.139193534851074, 0.0)),
    Vector((13.003655433654785, -0.3339431881904602, 0.0)),
    Vector((12.412906646728516, 2.05942440032959, 0.0)),
    Vector((11.421037673950195, 3.6512911319732666, 0.0)),
    Vector((10.137445449829102, 4.83127498626709, 0.0)),
    Vector((8.219351768493652, 5.699563980102539, 0.0)),
    Vector((5.644878387451172, 5.87767219543457, 0.0)),
    Vector((3.0266470909118652, 4.953718185424805, 0.0)),
    Vector((0.8460029363632202, 2.6716678142547607, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((0.4302951693534851, -4.575231075286865, 3.1181907653808594)),
    Vector((13.30997371673584, -3.139193534851074, 3.1181907653808594)),
    Vector((13.003655433654785, -0.3339431881904602, 3.1181907653808594)),
    Vector((12.412906646728516, 2.05942440032959, 3.1181907653808594)),
    Vector((11.421037673950195, 3.6512911319732666, 3.1181907653808594)),
    Vector((10.137445449829102, 4.83127498626709, 3.1181907653808594)),
    Vector((8.219351768493652, 5.699563980102539, 3.1181907653808594)),
    Vector((5.644878387451172, 5.87767219543457, 3.1181907653808594)),
    Vector((3.0266470909118652, 4.953718185424805, 3.1181907653808594)),
    Vector((0.8460029363632202, 2.6716678142547607, 3.1181907653808594)),
    Vector((0.0, 0.0, 3.1181907653808594))
]
unitVectors = [
    Vector((0.9938415884971619, 0.11080974340438843, 0.0)),
    Vector((-0.10854940116405487, 0.9940909743309021, 0.0)),
    Vector((-0.2396356165409088, 0.9708629250526428, 0.0)),
    Vector((-0.5288299322128296, 0.8487277626991272, 0.0)),
    Vector((-0.7361941337585449, 0.6767703890800476, 0.0)),
    Vector((-0.9110044836997986, 0.41239655017852783, 0.0)),
    Vector((-0.9976154565811157, 0.06901741772890091, 0.0)),
    Vector((-0.9430047273635864, -0.3327792286872864, 0.0)),
    Vector((-0.6908608078956604, -0.7229878306388855, 0.0)),
    Vector((-0.3018835484981537, -0.9533448219299316, 0.0)),
    Vector((0.09363564103841782, -0.995606541633606, 0.0))
]
holesInfo = None
firstVertIndex = 11
numPolygonVerts = 11
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
