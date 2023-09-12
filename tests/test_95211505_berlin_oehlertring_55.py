import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-0.5223356485366821, -3.539959669113159, 0.0)),
    Vector((-1.9265365600585938, -3.3284523487091064, 0.0)),
    Vector((-2.828756093978882, -9.10593318939209, 0.0)),
    Vector((-1.4245535135269165, -9.339705467224121, 0.0)),
    Vector((-2.2046682834625244, -14.204366683959961, 0.0)),
    Vector((5.616817951202393, -15.473405838012695, 0.0)),
    Vector((5.5557661056518555, -15.840760231018066, 0.0)),
    Vector((6.349446773529053, -15.974342346191406, 0.0)),
    Vector((6.552955150604248, -16.163585662841797, 0.0)),
    Vector((6.817515850067139, -16.36396026611328, 0.0)),
    Vector((7.244882583618164, -16.586599349975586, 0.0)),
    Vector((7.726518154144287, -16.753578186035156, 0.0)),
    Vector((8.153884887695312, -16.82036781311035, 0.0)),
    Vector((8.404878616333008, -16.831499099731445, 0.0)),
    Vector((8.900080680847168, -16.79810333251953, 0.0)),
    Vector((9.551305770874023, -16.619991302490234, 0.0)),
    Vector((9.917619705200195, -16.45301055908203, 0.0)),
    Vector((10.60954761505127, -16.55319595336914, 0.0)),
    Vector((12.725991249084473, -1.8256231546401978, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((-0.5223356485366821, -3.539959669113159, 5.688345909118652)),
    Vector((-1.9265365600585938, -3.3284523487091064, 5.688345909118652)),
    Vector((-2.828756093978882, -9.10593318939209, 5.688345909118652)),
    Vector((-1.4245535135269165, -9.339705467224121, 5.688345909118652)),
    Vector((-2.2046682834625244, -14.204366683959961, 5.688345909118652)),
    Vector((5.616817951202393, -15.473405838012695, 5.688345909118652)),
    Vector((5.5557661056518555, -15.840760231018066, 5.688345909118652)),
    Vector((6.349446773529053, -15.974342346191406, 5.688345909118652)),
    Vector((6.552955150604248, -16.163585662841797, 5.688345909118652)),
    Vector((6.817515850067139, -16.36396026611328, 5.688345909118652)),
    Vector((7.244882583618164, -16.586599349975586, 5.688345909118652)),
    Vector((7.726518154144287, -16.753578186035156, 5.688345909118652)),
    Vector((8.153884887695312, -16.82036781311035, 5.688345909118652)),
    Vector((8.404878616333008, -16.831499099731445, 5.688345909118652)),
    Vector((8.900080680847168, -16.79810333251953, 5.688345909118652)),
    Vector((9.551305770874023, -16.619991302490234, 5.688345909118652)),
    Vector((9.917619705200195, -16.45301055908203, 5.688345909118652)),
    Vector((10.60954761505127, -16.55319595336914, 5.688345909118652)),
    Vector((12.725991249084473, -1.8256231546401978, 5.688345909118652)),
    Vector((0.0, 0.0, 5.688345909118652)),
]
unitVectors = [
    Vector((-0.9888455271720886, 0.14894454181194305, 0.0)),
    Vector((-0.15429142117500305, -0.9880253672599792, 0.0)),
    Vector((0.9864237308502197, -0.16422027349472046, 0.0)),
    Vector((-0.15834057331085205, -0.987384557723999, 0.0)),
    Vector((0.9870916604995728, -0.1601559966802597, 0.0)),
    Vector((-0.16394464671611786, -0.986469566822052, 0.0)),
    Vector((0.9861302971839905, -0.1659727543592453, 0.0)),
    Vector((0.7323065400123596, -0.6809750199317932, 0.0)),
    Vector((0.7971649169921875, -0.6037616729736328, 0.0)),
    Vector((0.8868696689605713, -0.462019681930542, 0.0)),
    Vector((0.9448290467262268, -0.32756397128105164, 0.0)),
    Vector((0.9880073070526123, -0.1544075310230255, 0.0)),
    Vector((0.9990180730819702, -0.04430531710386276, 0.0)),
    Vector((0.9977337718009949, 0.06728583574295044, 0.0)),
    Vector((0.9645736217498779, 0.2638137936592102, 0.0)),
    Vector((0.9099220633506775, 0.4147793650627136, 0.0)),
    Vector((0.9896796941757202, -0.14329735934734344, 0.0)),
    Vector((0.14224493503570557, 0.9898315072059631, 0.0)),
    Vector((-0.989866316318512, 0.14200252294540405, 0.0)),
    Vector((-0.1459735929965973, -0.9892884492874146, 0.0)),
]
holesInfo = None
firstVertIndex = 20
numPolygonVerts = 20
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
