import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-25.161170959472656, -6.445351600646973, 0.0)),
    Vector((-22.704090118408203, -16.152420043945312, 0.0)),
    Vector((-19.794343948364258, -15.406588554382324, 0.0)),
    Vector((-18.09699249267578, -15.818475723266602, 0.0)),
    Vector((-16.351146697998047, -15.707159996032715, 0.0)),
    Vector((-14.330489158630371, -14.861136436462402, 0.0)),
    Vector((-13.021101951599121, -13.681153297424316, 0.0)),
    Vector((-7.21777868270874, -12.189480781555176, 0.0)),
    Vector((-5.075884819030762, -12.623628616333008, 0.0)),
    Vector((-3.3381216526031494, -12.38985824584961, 0.0)),
    Vector((-1.7458455562591553, -11.644018173217773, 0.0)),
    Vector((-0.4445439279079437, -10.464032173156738, 0.0)),
    Vector((2.4571151733398438, -9.718191146850586, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((-25.161170959472656, -6.445351600646973, 12.999817848205566)),
    Vector((-22.704090118408203, -16.152420043945312, 12.999817848205566)),
    Vector((-19.794343948364258, -15.406588554382324, 12.999817848205566)),
    Vector((-18.09699249267578, -15.818475723266602, 12.999817848205566)),
    Vector((-16.351146697998047, -15.707159996032715, 12.999817848205566)),
    Vector((-14.330489158630371, -14.861136436462402, 12.999817848205566)),
    Vector((-13.021101951599121, -13.681153297424316, 12.999817848205566)),
    Vector((-7.21777868270874, -12.189480781555176, 12.999817848205566)),
    Vector((-5.075884819030762, -12.623628616333008, 12.999817848205566)),
    Vector((-3.3381216526031494, -12.38985824584961, 12.999817848205566)),
    Vector((-1.7458455562591553, -11.644018173217773, 12.999817848205566)),
    Vector((-0.4445439279079437, -10.464032173156738, 12.999817848205566)),
    Vector((2.4571151733398438, -9.718191146850586, 12.999817848205566)),
    Vector((0.0, 0.0, 12.999817848205566))
]
unitVectors = [
    Vector((0.2453838735818863, -0.9694260358810425, 0.0)),
    Vector((0.9686844944953918, 0.24829497933387756, 0.0)),
    Vector((0.9717966318130493, -0.23582066595554352, 0.0)),
    Vector((0.9979734420776367, 0.06363113224506378, 0.0)),
    Vector((0.9224138259887695, 0.38620293140411377, 0.0)),
    Vector((0.7428610324859619, 0.6694456338882446, 0.0)),
    Vector((0.968517541885376, 0.2489454597234726, 0.0)),
    Vector((0.980069637298584, -0.1986536979675293, 0.0)),
    Vector((0.9910725951194763, 0.13332277536392212, 0.0)),
    Vector((0.905576765537262, 0.42418238520622253, 0.0)),
    Vector((0.7407939434051514, 0.6717324256896973, 0.0)),
    Vector((0.9685171246528625, 0.2489471733570099, 0.0)),
    Vector((-0.24512311816215515, 0.9694918990135193, 0.0)),
    Vector((-0.9687215089797974, -0.24815024435520172, 0.0))
]
holesInfo = None
firstVertIndex = 14
numPolygonVerts = 14
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
