import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-6.28377628326416, -5.810873508453369, 0.0)),
    Vector((-4.00776481628418, -8.271037101745605, 0.0)),
    Vector((-4.721670627593994, -9.028008460998535, 0.0)),
    Vector((-5.060952663421631, -9.517813682556152, 0.0)),
    Vector((-5.2023210525512695, -10.130070686340332, 0.0)),
    Vector((-5.06802225112915, -10.820252418518066, 0.0)),
    Vector((-4.820629596710205, -11.221002578735352, 0.0)),
    Vector((-4.248092174530029, -11.65514850616455, 0.0)),
    Vector((-3.583665132522583, -11.799864768981934, 0.0)),
    Vector((-2.8909645080566406, -11.644018173217773, 0.0)),
    Vector((-2.4032466411590576, -11.287796020507812, 0.0)),
    Vector((-1.8165714740753174, -10.675539016723633, 0.0)),
    Vector((0.40289735794067383, -13.04664421081543, 0.0)),
    Vector((6.686675071716309, -7.235762596130371, 0.0)),
    Vector((0.0, 0.0, 6.1665263175964355)),
    Vector((-6.28377628326416, -5.810873508453369, 6.1665263175964355)),
    Vector((-4.00776481628418, -8.271037101745605, 6.1665263175964355)),
    Vector((-4.721670627593994, -9.028008460998535, 6.1665263175964355)),
    Vector((-5.060952663421631, -9.517813682556152, 6.1665263175964355)),
    Vector((-5.2023210525512695, -10.130070686340332, 6.1665263175964355)),
    Vector((-5.06802225112915, -10.820252418518066, 6.1665263175964355)),
    Vector((-4.820629596710205, -11.221002578735352, 6.1665263175964355)),
    Vector((-4.248092174530029, -11.65514850616455, 6.1665263175964355)),
    Vector((-3.583665132522583, -11.799864768981934, 6.1665263175964355)),
    Vector((-2.8909645080566406, -11.644018173217773, 6.1665263175964355)),
    Vector((-2.4032466411590576, -11.287796020507812, 6.1665263175964355)),
    Vector((-1.8165714740753174, -10.675539016723633, 6.1665263175964355)),
    Vector((0.40289735794067383, -13.04664421081543, 6.1665263175964355)),
    Vector((6.686675071716309, -7.235762596130371, 6.1665263175964355))
]
unitVectors = [
    Vector((-0.7341938018798828, -0.6789399981498718, 0.0)),
    Vector((0.6790999174118042, -0.7340459227561951, 0.0)),
    Vector((-0.6861094832420349, -0.7274982929229736, 0.0)),
    Vector((-0.569421112537384, -0.8220460414886475, 0.0)),
    Vector((-0.22497782111167908, -0.9743638634681702, 0.0)),
    Vector((0.19100230932235718, -0.9815896153450012, 0.0)),
    Vector((0.52529376745224, -0.8509208559989929, 0.0)),
    Vector((0.7968204617500305, -0.6042161583900452, 0.0)),
    Vector((0.9770921468734741, -0.21281662583351135, 0.0)),
    Vector((0.9756130576133728, 0.21949738264083862, 0.0)),
    Vector((0.8075386881828308, 0.5898147225379944, 0.0)),
    Vector((0.6918615698814392, 0.7220300436019897, 0.0)),
    Vector((0.6833767890930176, -0.7300657629966736, 0.0)),
    Vector((0.7341933846473694, 0.67894047498703, 0.0)),
    Vector((-0.6786915063858032, 0.7344233989715576, 0.0))
]
holesInfo = None
firstVertIndex = 15
numPolygonVerts = 15
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
    assert not bpypolyskel.checkEdgeCrossing(bpypolyskel.debugOutputs["skeleton"])