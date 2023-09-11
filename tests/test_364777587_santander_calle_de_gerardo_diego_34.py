import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-5.494866847991943, -0.3005603849887848, 0.0)),
    Vector((-5.527188777923584, 0.30056488513946533, 0.0)),
    Vector((-8.113008499145508, 0.15585218369960785, 0.0)),
    Vector((-8.088767051696777, -0.44527310132980347, 0.0)),
    Vector((-15.26441764831543, -0.8460108041763306, 0.0)),
    Vector((-15.30482006072998, -0.24488547444343567, 0.0)),
    Vector((-17.890640258789062, -0.3895944356918335, 0.0)),
    Vector((-17.858318328857422, -0.9907197952270508, 0.0)),
    Vector((-29.922786712646484, -1.6697258949279785, 0.0)),
    Vector((-29.955106735229492, -1.0686004161834717, 0.0)),
    Vector((-32.549007415771484, -1.213303804397583, 0.0)),
    Vector((-32.51668930053711, -1.8144291639328003, 0.0)),
    Vector((-39.69234085083008, -2.2040088176727295, 0.0)),
    Vector((-39.73274230957031, -1.6028834581375122, 0.0)),
    Vector((-42.32664489746094, -1.7475829124450684, 0.0)),
    Vector((-42.2862434387207, -2.348708391189575, 0.0)),
    Vector((-48.2821159362793, -2.682626485824585, 0.0)),
    Vector((-47.95085144042969, -8.49350643157959, 0.0)),
    Vector((-48.573062896728516, -8.515766143798828, 0.0)),
    Vector((-48.37107467651367, -12.122518539428711, 0.0)),
    Vector((1.0100871324539185, -9.373101234436035, 0.0)),
    Vector((0.48484113812446594, 0.03339586406946182, 0.0)),
    Vector((-5.494866847991943, -0.3005603849887848, 14.810602188110352)),
    Vector((-5.527188777923584, 0.30056488513946533, 14.810602188110352)),
    Vector((-8.113008499145508, 0.15585218369960785, 14.810602188110352)),
    Vector((-8.088767051696777, -0.44527310132980347, 14.810602188110352)),
    Vector((-15.26441764831543, -0.8460108041763306, 14.810602188110352)),
    Vector((-15.30482006072998, -0.24488547444343567, 14.810602188110352)),
    Vector((-17.890640258789062, -0.3895944356918335, 14.810602188110352)),
    Vector((-17.858318328857422, -0.9907197952270508, 14.810602188110352)),
    Vector((-29.922786712646484, -1.6697258949279785, 14.810602188110352)),
    Vector((-29.955106735229492, -1.0686004161834717, 14.810602188110352)),
    Vector((-32.549007415771484, -1.213303804397583, 14.810602188110352)),
    Vector((-32.51668930053711, -1.8144291639328003, 14.810602188110352)),
    Vector((-39.69234085083008, -2.2040088176727295, 14.810602188110352)),
    Vector((-39.73274230957031, -1.6028834581375122, 14.810602188110352)),
    Vector((-42.32664489746094, -1.7475829124450684, 14.810602188110352)),
    Vector((-42.2862434387207, -2.348708391189575, 14.810602188110352)),
    Vector((-48.2821159362793, -2.682626485824585, 14.810602188110352)),
    Vector((-47.95085144042969, -8.49350643157959, 14.810602188110352)),
    Vector((-48.573062896728516, -8.515766143798828, 14.810602188110352)),
    Vector((-48.37107467651367, -12.122518539428711, 14.810602188110352)),
    Vector((1.0100871324539185, -9.373101234436035, 14.810602188110352)),
    Vector((0.48484113812446594, 0.03339586406946182, 14.810602188110352))
]
unitVectors = [
    Vector((-0.05369148775935173, 0.9985576272010803, 0.0)),
    Vector((-0.9984377026557922, -0.05587652325630188, 0.0)),
    Vector((0.04029402881860733, -0.9991878867149353, 0.0)),
    Vector((-0.998444139957428, -0.05575999245047569, 0.0)),
    Vector((-0.06705999374389648, 0.9977489113807678, 0.0)),
    Vector((-0.9984378218650818, -0.05587507411837578, 0.0)),
    Vector((0.053691476583480835, -0.9985575675964355, 0.0)),
    Vector((-0.998420000076294, -0.0561925508081913, 0.0)),
    Vector((-0.053688306361436844, 0.9985576868057251, 0.0)),
    Vector((-0.998447597026825, -0.055699415504932404, 0.0)),
    Vector((0.053685154765844345, -0.9985578656196594, 0.0)),
    Vector((-0.9985294938087463, -0.0542120486497879, 0.0)),
    Vector((-0.06705841422080994, 0.9977489709854126, 0.0)),
    Vector((-0.9984477162361145, -0.05569786578416824, 0.0)),
    Vector((0.06705840677022934, -0.9977490305900574, 0.0)),
    Vector((-0.99845290184021, -0.05560516566038132, 0.0)),
    Vector((0.056915223598480225, -0.9983790516853333, 0.0)),
    Vector((-0.9993607401847839, -0.03575228527188301, 0.0)),
    Vector((0.05591518059372902, -0.9984354972839355, 0.0)),
    Vector((0.9984536170959473, 0.05559135600924492, 0.0)),
    Vector((-0.05575179308652878, 0.998444676399231, 0.0)),
    Vector((-0.9984441995620728, -0.05576135963201523, 0.0))
]
holesInfo = None
firstVertIndex = 22
numPolygonVerts = 22
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
