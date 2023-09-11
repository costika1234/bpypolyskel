import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-4.9820942878723145, -7.8257575035095215, 0.0)),
    Vector((4.286461353302002, -13.725690841674805, 0.0)),
    Vector((4.1407084465026855, -13.95946216583252, 0.0)),
    Vector((19.9085693359375, -23.989307403564453, 0.0)),
    Vector((20.06757164001465, -23.744403839111328, 0.0)),
    Vector((23.433156967163086, -25.881723403930664, 0.0)),
    Vector((29.163856506347656, -16.875944137573242, 0.0)),
    Vector((26.06990623474121, -14.905608177185059, 0.0)),
    Vector((42.64575958251953, 11.14327335357666, 0.0)),
    Vector((26.92434310913086, 21.1396484375, 0.0)),
    Vector((14.595121383666992, 1.7700024843215942, 0.0)),
    Vector((5.008577823638916, 7.870290756225586, 0.0)),
    Vector((-4.9820942878723145, -7.8257575035095215, 18.409103393554688)),
    Vector((4.286461353302002, -13.725690841674805, 18.409103393554688)),
    Vector((4.1407084465026855, -13.95946216583252, 18.409103393554688)),
    Vector((19.9085693359375, -23.989307403564453, 18.409103393554688)),
    Vector((20.06757164001465, -23.744403839111328, 18.409103393554688)),
    Vector((23.433156967163086, -25.881723403930664, 18.409103393554688)),
    Vector((29.163856506347656, -16.875944137573242, 18.409103393554688)),
    Vector((26.06990623474121, -14.905608177185059, 18.409103393554688)),
    Vector((42.64575958251953, 11.14327335357666, 18.409103393554688)),
    Vector((26.92434310913086, 21.1396484375, 18.409103393554688)),
    Vector((14.595121383666992, 1.7700024843215942, 18.409103393554688)),
    Vector((5.008577823638916, 7.870290756225586, 18.409103393554688))
]
unitVectors = [
    Vector((0.8435888886451721, -0.5369896292686462, 0.0)),
    Vector((-0.5290741324424744, -0.8485755920410156, 0.0)),
    Vector((0.8437643647193909, -0.5367136597633362, 0.0)),
    Vector((0.5445428490638733, 0.8387330174446106, 0.0)),
    Vector((0.8441628813743591, -0.5360867977142334, 0.0)),
    Vector((0.5368587970733643, 0.8436721563339233, 0.0)),
    Vector((-0.8434813618659973, 0.5371584892272949, 0.0)),
    Vector((0.5368591547012329, 0.8436717987060547, 0.0)),
    Vector((-0.8438599109649658, 0.5365636348724365, 0.0)),
    Vector((-0.5369710326194763, -0.8436006307601929, 0.0)),
    Vector((-0.8436710834503174, 0.5368605256080627, 0.0)),
    Vector((-0.5369625687599182, -0.8436059951782227, 0.0))
]
holesInfo = None
firstVertIndex = 12
numPolygonVerts = 12
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
