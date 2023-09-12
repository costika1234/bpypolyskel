import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-2.6914963722229004, 0.18924367427825928, 0.0)),
    Vector((-4.938451766967773, -0.8794221878051758, 0.0)),
    Vector((-13.546390533447266, -0.3005490005016327, 0.0)),
    Vector((-15.63977336883545, 1.0464214086532593, 0.0)),
    Vector((-18.63032341003418, 1.235672116279602, 0.0)),
    Vector((-20.90961265563965, 0.17814365029335022, 0.0)),
    Vector((-29.485218048095703, 0.7459051609039307, 0.0)),
    Vector((-31.594762802124023, 2.0928804874420166, 0.0)),
    Vector((-34.58531188964844, 2.2932703495025635, 0.0)),
    Vector((-36.84843826293945, 1.2357470989227295, 0.0)),
    Vector((-41.140281677246094, 1.5252026319503784, 0.0)),
    Vector((-41.7950325012207, -8.482415199279785, 0.0)),
    Vector((-21.863388061523438, -9.796079635620117, 0.0)),
    Vector((-20.07714080810547, -11.031731605529785, 0.0)),
    Vector((-17.967588424682617, -11.566071510314941, 0.0)),
    Vector((-15.817619323730469, -11.321173667907715, 0.0)),
    Vector((-13.885879516601562, -10.330434799194336, 0.0)),
    Vector((-5.908369541168213, -10.85364818572998, 0.0)),
    Vector((-4.13020133972168, -12.089295387268066, 0.0)),
    Vector((-2.028729200363159, -12.623629570007324, 0.0)),
    Vector((0.1293213814496994, -12.378726959228516, 0.0)),
    Vector((2.052976608276367, -11.387983322143555, 0.0)),
    Vector((6.037691593170166, -11.64401626586914, 0.0)),
    Vector((6.700453758239746, -1.6475250720977783, 0.0)),
    Vector((2.384359121322632, -1.3469654321670532, 0.0)),
    Vector((0.29905515909194946, -0.011131941340863705, 0.0)),
    Vector((-2.6914963722229004, 0.18924367427825928, 12.53545093536377)),
    Vector((-4.938451766967773, -0.8794221878051758, 12.53545093536377)),
    Vector((-13.546390533447266, -0.3005490005016327, 12.53545093536377)),
    Vector((-15.63977336883545, 1.0464214086532593, 12.53545093536377)),
    Vector((-18.63032341003418, 1.235672116279602, 12.53545093536377)),
    Vector((-20.90961265563965, 0.17814365029335022, 12.53545093536377)),
    Vector((-29.485218048095703, 0.7459051609039307, 12.53545093536377)),
    Vector((-31.594762802124023, 2.0928804874420166, 12.53545093536377)),
    Vector((-34.58531188964844, 2.2932703495025635, 12.53545093536377)),
    Vector((-36.84843826293945, 1.2357470989227295, 12.53545093536377)),
    Vector((-41.140281677246094, 1.5252026319503784, 12.53545093536377)),
    Vector((-41.7950325012207, -8.482415199279785, 12.53545093536377)),
    Vector((-21.863388061523438, -9.796079635620117, 12.53545093536377)),
    Vector((-20.07714080810547, -11.031731605529785, 12.53545093536377)),
    Vector((-17.967588424682617, -11.566071510314941, 12.53545093536377)),
    Vector((-15.817619323730469, -11.321173667907715, 12.53545093536377)),
    Vector((-13.885879516601562, -10.330434799194336, 12.53545093536377)),
    Vector((-5.908369541168213, -10.85364818572998, 12.53545093536377)),
    Vector((-4.13020133972168, -12.089295387268066, 12.53545093536377)),
    Vector((-2.028729200363159, -12.623629570007324, 12.53545093536377)),
    Vector((0.1293213814496994, -12.378726959228516, 12.53545093536377)),
    Vector((2.052976608276367, -11.387983322143555, 12.53545093536377)),
    Vector((6.037691593170166, -11.64401626586914, 12.53545093536377)),
    Vector((6.700453758239746, -1.6475250720977783, 12.53545093536377)),
    Vector((2.384359121322632, -1.3469654321670532, 12.53545093536377)),
    Vector((0.29905515909194946, -0.011131941340863705, 12.53545093536377)),
]
unitVectors = [
    Vector((-0.9030652046203613, -0.4295033812522888, 0.0)),
    Vector((-0.9977464079856873, 0.06709720194339752, 0.0)),
    Vector((-0.8409546613693237, 0.5411055684089661, 0.0)),
    Vector((-0.9980036616325378, 0.06315657496452332, 0.0)),
    Vector((-0.9071173071861267, -0.4208778738975525, 0.0)),
    Vector((-0.9978155493736267, 0.06606195867061615, 0.0)),
    Vector((-0.8428393006324768, 0.538165271282196, 0.0)),
    Vector((-0.9977624416351318, 0.06685778498649597, 0.0)),
    Vector((-0.9059686064720154, -0.42334482073783875, 0.0)),
    Vector((-0.9977334141731262, 0.0672903060913086, 0.0)),
    Vector((-0.06528566777706146, -0.9978666305541992, 0.0)),
    Vector((0.997835099697113, -0.06576579809188843, 0.0)),
    Vector((0.8224034309387207, -0.56890469789505, 0.0)),
    Vector((0.9693862795829773, -0.24554108083248138, 0.0)),
    Vector((0.9935749769210815, 0.11317574977874756, 0.0)),
    Vector((0.8897981643676758, 0.4563542306423187, 0.0)),
    Vector((0.997856080532074, -0.0654454454779625, 0.0)),
    Vector((0.8211950063705444, -0.5706475377082825, 0.0)),
    Vector((0.9691617488861084, -0.24642546474933624, 0.0)),
    Vector((0.9936223030090332, 0.11275950074195862, 0.0)),
    Vector((0.8890179395675659, 0.45787250995635986, 0.0)),
    Vector((0.9979421496391296, -0.0641215369105339, 0.0)),
    Vector((0.06615424156188965, 0.9978094100952148, 0.0)),
    Vector((-0.9975841045379639, 0.06946871429681778, 0.0)),
    Vector((-0.8420441746711731, 0.5394085645675659, 0.0)),
    Vector((-0.9977627992630005, 0.06685300171375275, 0.0)),
]
holesInfo = None
firstVertIndex = 26
numPolygonVerts = 26
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
