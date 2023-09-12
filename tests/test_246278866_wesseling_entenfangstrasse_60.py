import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((5.590058326721191, -6.411999702453613, 0.0)),
    Vector((22.887531280517578, 8.805421829223633, 0.0)),
    Vector((17.381845474243164, 15.128347396850586, 0.0)),
    Vector((14.970049858093262, 12.979874610900879, 0.0)),
    Vector((14.484875679016113, 13.514205932617188, 0.0)),
    Vector((13.795790672302246, 12.924211502075195, 0.0)),
    Vector((14.055956840515137, 12.657045364379883, 0.0)),
    Vector((14.3020601272583, 12.256295204162598, 0.0)),
    Vector((14.37940788269043, 11.888941764831543, 0.0)),
    Vector((14.351283073425293, 11.532718658447266, 0.0)),
    Vector((14.23878002166748, 11.187628746032715, 0.0)),
    Vector((14.013772964477539, 10.864801406860352, 0.0)),
    Vector((13.746576309204102, 10.64216136932373, 0.0)),
    Vector((13.486411094665527, 10.508577346801758, 0.0)),
    Vector((13.064521789550781, 10.430652618408203, 0.0)),
    Vector((12.670758247375488, 10.4640474319458, 0.0)),
    Vector((12.361371994018555, 10.586498260498047, 0.0)),
    Vector((12.101205825805664, 10.764608383178711, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((5.590058326721191, -6.411999702453613, 6.0)),
    Vector((22.887531280517578, 8.805421829223633, 6.0)),
    Vector((17.381845474243164, 15.128347396850586, 6.0)),
    Vector((14.970049858093262, 12.979874610900879, 6.0)),
    Vector((14.484875679016113, 13.514205932617188, 6.0)),
    Vector((13.795790672302246, 12.924211502075195, 6.0)),
    Vector((14.055956840515137, 12.657045364379883, 6.0)),
    Vector((14.3020601272583, 12.256295204162598, 6.0)),
    Vector((14.37940788269043, 11.888941764831543, 6.0)),
    Vector((14.351283073425293, 11.532718658447266, 6.0)),
    Vector((14.23878002166748, 11.187628746032715, 6.0)),
    Vector((14.013772964477539, 10.864801406860352, 6.0)),
    Vector((13.746576309204102, 10.64216136932373, 6.0)),
    Vector((13.486411094665527, 10.508577346801758, 6.0)),
    Vector((13.064521789550781, 10.430652618408203, 6.0)),
    Vector((12.670758247375488, 10.4640474319458, 6.0)),
    Vector((12.361371994018555, 10.586498260498047, 6.0)),
    Vector((12.101205825805664, 10.764608383178711, 6.0)),
    Vector((0.0, 0.0, 6.0)),
]
unitVectors = [
    Vector((0.7508072257041931, 0.6605213284492493, 0.0)),
    Vector((-0.6566872596740723, 0.7541630268096924, 0.0)),
    Vector((-0.7466933727264404, -0.6651684641838074, 0.0)),
    Vector((-0.6722314953804016, 0.7403410077095032, 0.0)),
    Vector((-0.7596104741096497, -0.6503783464431763, 0.0)),
    Vector((0.6976589560508728, -0.7164300084114075, 0.0)),
    Vector((0.523307204246521, -0.8521440625190735, 0.0)),
    Vector((0.20603646337985992, -0.978544294834137, 0.0)),
    Vector((-0.07870785892009735, -0.9968976974487305, 0.0)),
    Vector((-0.3099551796913147, -0.9507511258125305, 0.0)),
    Vector((-0.5718032121658325, -0.8203907608985901, 0.0)),
    Vector((-0.7682549953460693, -0.6401439309120178, 0.0)),
    Vector((-0.8895869255065918, -0.45676589012145996, 0.0)),
    Vector((-0.9833665490150452, -0.181631937623024, 0.0)),
    Vector((-0.9964230060577393, 0.08450594544410706, 0.0)),
    Vector((-0.9298216104507446, 0.36801061034202576, 0.0)),
    Vector((-0.8251569271087646, 0.5649036169052124, 0.0)),
    Vector((-0.7471646070480347, -0.6646390557289124, 0.0)),
    Vector((0.6571425199508667, -0.7537662982940674, 0.0)),
]
holesInfo = None
firstVertIndex = 19
numPolygonVerts = 19
faces = []

bpypolyskel.debug_outputs["skeleton"] = 1


@pytest.mark.dependency()
@pytest.mark.timeout(10)
def test_polygonize():
    global faces
    faces = bpypolyskel.polygonize(
        verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, unitVectors
    )


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
