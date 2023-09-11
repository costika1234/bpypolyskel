import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-4.212701320648193, -6.635931491851807, 0.0)),
    Vector((7.417003154754639, -14.018830299377441, 0.0)),
    Vector((11.68514633178711, -7.295539855957031, 0.0)),
    Vector((6.454980850219727, -3.975275993347168, 0.0)),
    Vector((5.168192386627197, -6.002255916595459, 0.0)),
    Vector((7.1271514892578125, -7.24586296081543, 0.0)),
    Vector((5.762024402618408, -9.39624309539795, 0.0)),
    Vector((0.32220786809921265, -5.942880153656006, 0.0)),
    Vector((1.764411449432373, -3.671090602874756, 0.0)),
    Vector((3.9966232776641846, -5.0881667137146, 0.0)),
    Vector((5.150887489318848, -3.2699434757232666, 0.0)),
    Vector((0.0, 0.0, 14.22441291809082)),
    Vector((-4.212701320648193, -6.635931491851807, 14.22441291809082)),
    Vector((7.417003154754639, -14.018830299377441, 14.22441291809082)),
    Vector((11.68514633178711, -7.295539855957031, 14.22441291809082)),
    Vector((6.454980850219727, -3.975275993347168, 14.22441291809082)),
    Vector((5.168192386627197, -6.002255916595459, 14.22441291809082)),
    Vector((7.1271514892578125, -7.24586296081543, 14.22441291809082)),
    Vector((5.762024402618408, -9.39624309539795, 14.22441291809082)),
    Vector((0.32220786809921265, -5.942880153656006, 14.22441291809082)),
    Vector((1.764411449432373, -3.671090602874756, 14.22441291809082)),
    Vector((3.9966232776641846, -5.0881667137146, 14.22441291809082)),
    Vector((5.150887489318848, -3.2699434757232666, 14.22441291809082))
]
unitVectors = [
    Vector((-0.5359547734260559, -0.8442466855049133, 0.0)),
    Vector((0.8442469835281372, -0.5359542965888977, 0.0)),
    Vector((0.5359533429145813, 0.8442476987838745, 0.0)),
    Vector((-0.844247579574585, 0.5359533429145813, 0.0)),
    Vector((-0.5359538197517395, -0.8442472219467163, 0.0)),
    Vector((0.8442472815513611, -0.535953938961029, 0.0)),
    Vector((-0.535953938961029, -0.8442471623420715, 0.0)),
    Vector((-0.8442471623420715, 0.5359540581703186, 0.0)),
    Vector((0.535954475402832, 0.8442468047142029, 0.0)),
    Vector((0.8442473411560059, -0.535953938961029, 0.0)),
    Vector((0.5359540581703186, 0.844247043132782, 0.0)),
    Vector((-0.844247043132782, 0.5359542965888977, 0.0))
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
