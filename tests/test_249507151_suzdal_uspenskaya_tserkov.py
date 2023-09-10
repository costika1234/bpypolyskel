import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((5.0059895515441895, 3.996372699737549, 0.0)),
    Vector((-1.8102803230285645, 12.968721389770508, 0.0)),
    Vector((1.5824549198150635, 15.5513334274292, 0.0)),
    Vector((1.6317138671875, 16.4418888092041, 0.0)),
    Vector((1.3423153162002563, 17.39923667907715, 0.0)),
    Vector((0.9667131304740906, 18.067153930664062, 0.0)),
    Vector((0.09851852059364319, 18.757333755493164, 0.0)),
    Vector((-0.7265740633010864, 19.013368606567383, 0.0)),
    Vector((-1.582453727722168, 18.824125289916992, 0.0)),
    Vector((-9.408531188964844, 13.202502250671387, 0.0)),
    Vector((0.0, 0.0, 2.986158847808838)),
    Vector((5.0059895515441895, 3.996372699737549, 2.986158847808838)),
    Vector((-1.8102803230285645, 12.968721389770508, 2.986158847808838)),
    Vector((1.5824549198150635, 15.5513334274292, 2.986158847808838)),
    Vector((1.6317138671875, 16.4418888092041, 2.986158847808838)),
    Vector((1.3423153162002563, 17.39923667907715, 2.986158847808838)),
    Vector((0.9667131304740906, 18.067153930664062, 2.986158847808838)),
    Vector((0.09851852059364319, 18.757333755493164, 2.986158847808838)),
    Vector((-0.7265740633010864, 19.013368606567383, 2.986158847808838)),
    Vector((-1.582453727722168, 18.824125289916992, 2.986158847808838)),
    Vector((-9.408531188964844, 13.202502250671387, 2.986158847808838))
]
unitVectors = [
    Vector((0.781509518623352, 0.6238932609558105, 0.0)),
    Vector((-0.6049304604530334, 0.7962782382965088, 0.0)),
    Vector((0.7956949472427368, 0.6056975722312927, 0.0)),
    Vector((0.05522819235920906, 0.9984737038612366, 0.0)),
    Vector((-0.289359986782074, 0.9572202563285828, 0.0)),
    Vector((-0.49016088247299194, 0.8716320395469666, 0.0)),
    Vector((-0.7827897071838379, 0.6222863793373108, 0.0)),
    Vector((-0.9550734758377075, 0.2963692843914032, 0.0)),
    Vector((-0.9764164686203003, -0.2158951759338379, 0.0)),
    Vector((-0.8121810555458069, -0.5834053754806519, 0.0)),
    Vector((0.5803462862968445, -0.814369797706604, 0.0))
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
