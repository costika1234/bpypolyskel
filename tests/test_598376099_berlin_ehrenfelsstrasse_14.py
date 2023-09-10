import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.35932549834251404, 0.5120696425437927, 0.0)),
    Vector((-19.084875106811523, 14.326855659484863, 0.0)),
    Vector((-25.79683494567871, 5.0539727210998535, 0.0)),
    Vector((-26.04090690612793, 4.608695983886719, 0.0)),
    Vector((-18.156112670898438, -0.9795778393745422, 0.0)),
    Vector((-19.118839263916016, -2.348803997039795, 0.0)),
    Vector((-15.552708625793457, -4.8646368980407715, 0.0)),
    Vector((-14.549304008483887, -3.45088267326355, 0.0)),
    Vector((-6.488210678100586, -9.161589622497559, 0.0)),
    Vector((0.35932549834251404, 0.5120696425437927, 9.953322410583496)),
    Vector((-19.084875106811523, 14.326855659484863, 9.953322410583496)),
    Vector((-25.79683494567871, 5.0539727210998535, 9.953322410583496)),
    Vector((-26.04090690612793, 4.608695983886719, 9.953322410583496)),
    Vector((-18.156112670898438, -0.9795778393745422, 9.953322410583496)),
    Vector((-19.118839263916016, -2.348803997039795, 9.953322410583496)),
    Vector((-15.552708625793457, -4.8646368980407715, 9.953322410583496)),
    Vector((-14.549304008483887, -3.45088267326355, 9.953322410583496)),
    Vector((-6.488210678100586, -9.161589622497559, 9.953322410583496))
]
unitVectors = [
    Vector((-0.8151968121528625, 0.5791839957237244, 0.0)),
    Vector((-0.5863444209098816, -0.8100618720054626, 0.0)),
    Vector((-0.4806629717350006, -0.8769054412841797, 0.0)),
    Vector((0.8158677220344543, -0.5782386064529419, 0.0)),
    Vector((-0.5751725435256958, -0.8180321455001831, 0.0)),
    Vector((0.8171228170394897, -0.5764635801315308, 0.0)),
    Vector((0.5787835121154785, 0.8154812455177307, 0.0)),
    Vector((0.8159879446029663, -0.5780689716339111, 0.0)),
    Vector((0.5777566432952881, 0.8162090182304382, 0.0))
]
holesInfo = None
firstVertIndex = 9
numPolygonVerts = 9
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
