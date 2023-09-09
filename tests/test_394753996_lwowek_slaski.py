import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-2.704775094985962, -10.107809066772461, 0.0)),
    Vector((-2.7956342697143555, -11.95571231842041, 0.0)),
    Vector((-2.4322023391723633, -13.157962799072266, 0.0)),
    Vector((-1.9639338254928589, -13.937199592590332, 0.0)),
    Vector((-1.2999706268310547, -14.560588836669922, 0.0)),
    Vector((-0.552138090133667, -15.028131484985352, 0.0)),
    Vector((0.5241817831993103, -15.373221397399902, 0.0)),
    Vector((1.7053380012512207, -15.284165382385254, 0.0)),
    Vector((2.760690450668335, -15.117186546325684, 0.0)),
    Vector((3.634326219558716, -14.393609046936035, 0.0)),
    Vector((4.235387325286865, -13.580976486206055, 0.0)),
    Vector((5.290736675262451, -10.953835487365723, 0.0)),
    Vector((7.736903667449951, -2.515814781188965, 0.0)),
    Vector((0.0, 0.0, 2.956062078475952)),
    Vector((-2.704775094985962, -10.107809066772461, 2.956062078475952)),
    Vector((-2.7956342697143555, -11.95571231842041, 2.956062078475952)),
    Vector((-2.4322023391723633, -13.157962799072266, 2.956062078475952)),
    Vector((-1.9639338254928589, -13.937199592590332, 2.956062078475952)),
    Vector((-1.2999706268310547, -14.560588836669922, 2.956062078475952)),
    Vector((-0.552138090133667, -15.028131484985352, 2.956062078475952)),
    Vector((0.5241817831993103, -15.373221397399902, 2.956062078475952)),
    Vector((1.7053380012512207, -15.284165382385254, 2.956062078475952)),
    Vector((2.760690450668335, -15.117186546325684, 2.956062078475952)),
    Vector((3.634326219558716, -14.393609046936035, 2.956062078475952)),
    Vector((4.235387325286865, -13.580976486206055, 2.956062078475952)),
    Vector((5.290736675262451, -10.953835487365723, 2.956062078475952)),
    Vector((7.736903667449951, -2.515814781188965, 2.956062078475952))
]
unitVectors = [
    Vector((-0.2584976553916931, -0.9660118818283081, 0.0)),
    Vector((-0.04910946264863014, -0.9987933039665222, 0.0)),
    Vector((0.2893609404563904, -0.9572200775146484, 0.0)),
    Vector((0.5150831937789917, -0.8571402430534363, 0.0)),
    Vector((0.7290309071540833, -0.6844807267189026, 0.0)),
    Vector((0.847923219203949, -0.5301190614700317, 0.0)),
    Vector((0.9522526264190674, -0.3053114414215088, 0.0)),
    Vector((0.9971697330474854, 0.0751839280128479, 0.0)),
    Vector((0.9877132773399353, 0.1562768965959549, 0.0)),
    Vector((0.7701480388641357, 0.6378651261329651, 0.0)),
    Vector((0.5946595668792725, 0.8039777278900146, 0.0)),
    Vector((0.37275832891464233, 0.9279285073280334, 0.0)),
    Vector((0.27843424677848816, 0.9604552388191223, 0.0)),
    Vector((-0.9509863257408142, 0.3092329502105713, 0.0))
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
