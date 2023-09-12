import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 7.081154551613622e-10, 0.0)),
    Vector((-16.724899291992188, 14.427030563354492, 0.0)),
    Vector((-28.476409912109375, 0.8794965147972107, 0.0)),
    Vector((-22.38796615600586, -4.408206939697266, 0.0)),
    Vector((-50.43923568725586, -36.746337890625, 0.0)),
    Vector((-55.63275909423828, -32.23784637451172, 0.0)),
    Vector((-55.30265426635742, -31.692384719848633, 0.0)),
    Vector((-55.15593719482422, -30.92428207397461, 0.0)),
    Vector((-55.28063201904297, -30.089384078979492, 0.0)),
    Vector((-55.47868728637695, -29.699764251708984, 0.0)),
    Vector((-56.06552505493164, -29.087501525878906, 0.0)),
    Vector((-56.76239776611328, -28.74240493774414, 0.0)),
    Vector((-57.53262710571289, -28.65334129333496, 0.0)),
    Vector((-58.21483612060547, -28.8203125, 0.0)),
    Vector((-58.95573043823242, -29.36577033996582, 0.0)),
    Vector((-71.56536865234375, -17.498966217041016, 0.0)),
    Vector((-83.31707000732422, -30.38960075378418, 0.0)),
    Vector((-66.82695770263672, -45.596065521240234, 0.0)),
    Vector((-65.06639862060547, -43.570068359375, 0.0)),
    Vector((-47.90865707397461, -58.442527770996094, 0.0)),
    Vector((-46.338829040527344, -56.639163970947266, 0.0)),
    Vector((-42.76642608642578, -59.7338752746582, 0.0)),
    Vector((-35.4968147277832, -51.35157012939453, 0.0)),
    Vector((-41.37990188598633, -46.25309371948242, 0.0)),
    Vector((-9.961627960205078, -10.007613182067871, 0.0)),
    Vector((-8.729263305664062, -11.131941795349121, 0.0)),
    Vector((-7.504233360290527, -11.777597427368164, 0.0)),
    Vector((-6.4845967292785645, -12.122689247131348, 0.0)),
    Vector((-4.019863128662109, -12.278538703918457, 0.0)),
    Vector((-1.6284846067428589, -11.577226638793945, 0.0)),
    Vector((0.5428280234336853, -9.907434463500977, 0.0)),
    Vector((1.8338778018951416, -7.781231880187988, 0.0)),
    Vector((2.3106849193573, -5.354466915130615, 0.0)),
    Vector((1.9365732669830322, -2.9499661922454834, 0.0)),
    Vector((1.2543710470199585, -1.5696046352386475, 0.0)),
    Vector((0.7922342419624329, -0.8460280895233154, 0.0)),
    Vector((0.0, 7.081154551613622e-10, 6.508872032165527)),
    Vector((-16.724899291992188, 14.427030563354492, 6.508872032165527)),
    Vector((-28.476409912109375, 0.8794965147972107, 6.508872032165527)),
    Vector((-22.38796615600586, -4.408206939697266, 6.508872032165527)),
    Vector((-50.43923568725586, -36.746337890625, 6.508872032165527)),
    Vector((-55.63275909423828, -32.23784637451172, 6.508872032165527)),
    Vector((-55.30265426635742, -31.692384719848633, 6.508872032165527)),
    Vector((-55.15593719482422, -30.92428207397461, 6.508872032165527)),
    Vector((-55.28063201904297, -30.089384078979492, 6.508872032165527)),
    Vector((-55.47868728637695, -29.699764251708984, 6.508872032165527)),
    Vector((-56.06552505493164, -29.087501525878906, 6.508872032165527)),
    Vector((-56.76239776611328, -28.74240493774414, 6.508872032165527)),
    Vector((-57.53262710571289, -28.65334129333496, 6.508872032165527)),
    Vector((-58.21483612060547, -28.8203125, 6.508872032165527)),
    Vector((-58.95573043823242, -29.36577033996582, 6.508872032165527)),
    Vector((-71.56536865234375, -17.498966217041016, 6.508872032165527)),
    Vector((-83.31707000732422, -30.38960075378418, 6.508872032165527)),
    Vector((-66.82695770263672, -45.596065521240234, 6.508872032165527)),
    Vector((-65.06639862060547, -43.570068359375, 6.508872032165527)),
    Vector((-47.90865707397461, -58.442527770996094, 6.508872032165527)),
    Vector((-46.338829040527344, -56.639163970947266, 6.508872032165527)),
    Vector((-42.76642608642578, -59.7338752746582, 6.508872032165527)),
    Vector((-35.4968147277832, -51.35157012939453, 6.508872032165527)),
    Vector((-41.37990188598633, -46.25309371948242, 6.508872032165527)),
    Vector((-9.961627960205078, -10.007613182067871, 6.508872032165527)),
    Vector((-8.729263305664062, -11.131941795349121, 6.508872032165527)),
    Vector((-7.504233360290527, -11.777597427368164, 6.508872032165527)),
    Vector((-6.4845967292785645, -12.122689247131348, 6.508872032165527)),
    Vector((-4.019863128662109, -12.278538703918457, 6.508872032165527)),
    Vector((-1.6284846067428589, -11.577226638793945, 6.508872032165527)),
    Vector((0.5428280234336853, -9.907434463500977, 6.508872032165527)),
    Vector((1.8338778018951416, -7.781231880187988, 6.508872032165527)),
    Vector((2.3106849193573, -5.354466915130615, 6.508872032165527)),
    Vector((1.9365732669830322, -2.9499661922454834, 6.508872032165527)),
    Vector((1.2543710470199585, -1.5696046352386475, 6.508872032165527)),
    Vector((0.7922342419624329, -0.8460280895233154, 6.508872032165527)),
]
unitVectors = [
    Vector((-0.7572081685066223, 0.6531737446784973, 0.0)),
    Vector((-0.6552588939666748, -0.7554043531417847, 0.0)),
    Vector((0.7550103068351746, -0.6557128429412842, 0.0)),
    Vector((-0.6552624702453613, -0.7554012537002563, 0.0)),
    Vector((-0.7551535367965698, 0.6555479168891907, 0.0)),
    Vector((0.5177533030509949, 0.8555299639701843, 0.0)),
    Vector((0.18762025237083435, 0.9822416305541992, 0.0)),
    Vector((-0.14771494269371033, 0.9890299439430237, 0.0)),
    Vector((-0.4531439542770386, 0.8914374113082886, 0.0)),
    Vector((-0.6919581890106201, 0.7219375371932983, 0.0)),
    Vector((-0.8961385488510132, 0.4437745213508606, 0.0)),
    Vector((-0.9933807849884033, 0.11486723273992538, 0.0)),
    Vector((-0.9713303446769714, -0.2377338856458664, 0.0)),
    Vector((-0.805296778678894, -0.592871904373169, 0.0)),
    Vector((-0.7282313704490662, 0.6853312253952026, 0.0)),
    Vector((-0.6737061738967896, -0.7389993667602539, 0.0)),
    Vector((0.7351406216621399, -0.6779146790504456, 0.0)),
    Vector((0.655928909778595, 0.7548227310180664, 0.0)),
    Vector((0.7556363940238953, -0.6549912691116333, 0.0)),
    Vector((0.6565800309181213, 0.7542563080787659, 0.0)),
    Vector((0.7558326721191406, -0.6547648310661316, 0.0)),
    Vector((0.6551851034164429, 0.7554683685302734, 0.0)),
    Vector((-0.7557017803192139, 0.654915988445282, 0.0)),
    Vector((0.654996395111084, 0.7556321620941162, 0.0)),
    Vector((0.7387462854385376, -0.6739835739135742, 0.0)),
    Vector((0.8846492767333984, -0.4662570059299469, 0.0)),
    Vector((0.947220504283905, -0.32058289647102356, 0.0)),
    Vector((0.9980068802833557, -0.0631057396531105, 0.0)),
    Vector((0.9595862030982971, 0.28141483664512634, 0.0)),
    Vector((0.7927030324935913, 0.6096079349517822, 0.0)),
    Vector((0.5190195441246033, 0.8547623753547668, 0.0)),
    Vector((0.19279246032238007, 0.9812394976615906, 0.0)),
    Vector((-0.1537383794784546, 0.9881115555763245, 0.0)),
    Vector((-0.4430633783340454, 0.8964902758598328, 0.0)),
    Vector((-0.538266658782959, 0.8427745699882507, 0.0)),
    Vector((-0.6835198998451233, 0.7299318909645081, 0.0)),
]
holesInfo = None
firstVertIndex = 36
numPolygonVerts = 36
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
