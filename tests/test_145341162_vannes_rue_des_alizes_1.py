import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((0.7500497698783875, 4.815185050688342e-08, 0.0)),
    Vector((0.7500501871109009, -3.450904130935669, 0.0)),
    Vector((0.0, -3.3395848274230957, 0.0)),
    Vector((0.0, -4.230140686035156, 0.0)),
    Vector((0.6750452518463135, -4.230140686035156, 0.0)),
    Vector((0.7500508427619934, -8.460281372070312, 0.0)),
    Vector((0.0, -8.571600914001465, 0.0)),
    Vector((0.0, -9.239518165588379, 0.0)),
    Vector((0.7500509023666382, -9.128198623657227, 0.0)),
    Vector((0.6750462651252747, -12.690422058105469, 0.0)),
    Vector((0.07500513643026352, -12.690422058105469, 0.0)),
    Vector((0.07500514388084412, -13.358339309692383, 0.0)),
    Vector((0.7500514984130859, -13.469657897949219, 0.0)),
    Vector((0.6750465035438538, -15.139451026916504, 0.0)),
    Vector((2.550175666809082, -15.028130531311035, 0.0)),
    Vector((2.475170135498047, -13.803616523742676, 0.0)),
    Vector((11.100762367248535, -13.803606033325195, 0.0)),
    Vector((11.100753784179688, -9.350826263427734, 0.0)),
    Vector((12.150825500488281, -9.23950481414795, 0.0)),
    Vector((12.825871467590332, -9.239503860473633, 0.0)),
    Vector((12.900873184204102, -7.792349815368652, 0.0)),
    Vector((13.350902557373047, -7.235751628875732, 0.0)),
    Vector((13.72592544555664, -6.456514358520508, 0.0)),
    Vector((13.950936317443848, -4.786721229553223, 0.0)),
    Vector((13.125879287719727, -4.007486820220947, 0.0)),
    Vector((13.200883865356445, -3.7848477363586426, 0.0)),
    Vector((12.900863647460938, -3.78484845161438, 0.0)),
    Vector((12.82585620880127, -2.5603342056274414, 0.0)),
    Vector((11.025735855102539, -2.3376989364624023, 0.0)),
    Vector((10.950722694396973, 2.1150805950164795, 0.0)),
    Vector((9.375617980957031, 2.1150779724121094, 0.0)),
    Vector((9.525615692138672, 9.573484420776367, 0.0)),
    Vector((10.800698280334473, 9.573486328125, 0.0)),
    Vector((10.57567024230957, 17.031890869140625, 0.0)),
    Vector((9.975628852844238, 19.369600296020508, 0.0)),
    Vector((9.000564575195312, 21.150711059570312, 0.0)),
    Vector((7.5004682540893555, 22.82050132751465, 0.0)),
    Vector((7.4254631996154785, 22.931819915771484, 0.0)),
    Vector((5.400335788726807, 24.045013427734375, 0.0)),
    Vector((3.3752095699310303, 24.824247360229492, 0.0)),
    Vector((1.0500651597976685, 25.158205032348633, 0.0)),
    Vector((-6.075376510620117, 25.380847930908203, 0.0)),
    Vector((-6.1503825187683105, 23.933692932128906, 0.0)),
    Vector((-13.200820922851562, 23.933706283569336, 0.0)),
    Vector((-13.125811576843262, 26.271413803100586, 0.0)),
    Vector((-15.075931549072266, 26.271419525146484, 0.0)),
    Vector((-15.07593059539795, 26.605377197265625, 0.0)),
    Vector((-15.375947952270508, 27.161975860595703, 0.0)),
    Vector((-16.050987243652344, 27.82989501953125, 0.0)),
    Vector((-16.726028442382812, 27.94121551513672, 0.0)),
    Vector((-17.851099014282227, 27.495941162109375, 0.0)),
    Vector((-18.376134872436523, 26.716707229614258, 0.0)),
    Vector((-18.451141357421875, 26.160110473632812, 0.0)),
    Vector((-18.901168823242188, 26.160110473632812, 0.0)),
    Vector((-18.976179122924805, 24.378999710083008, 0.0)),
    Vector((-24.376514434814453, 24.379018783569336, 0.0)),
    Vector((-24.45155143737793, 16.697975158691406, 0.0)),
    Vector((-25.126596450805664, 16.141380310058594, 0.0)),
    Vector((-22.12641716003418, 13.024422645568848, 0.0)),
    Vector((-18.526187896728516, 13.024410247802734, 0.0)),
    Vector((-18.45119285583496, 9.796144485473633, 0.0)),
    Vector((-13.65088176727295, 9.907450675964355, 0.0)),
    Vector((-13.650884628295898, 8.682936668395996, 0.0)),
    Vector((-12.900835990905762, 8.682934761047363, 0.0)),
    Vector((-12.975838661193848, 9.684810638427734, 0.0)),
    Vector((-9.675625801086426, 9.68480396270752, 0.0)),
    Vector((-9.675626754760742, 8.7942476272583, 0.0)),
    Vector((-8.850573539733887, 8.794246673583984, 0.0)),
    Vector((-8.700562477111816, 9.684802055358887, 0.0)),
    Vector((-5.025324821472168, 9.796117782592773, 0.0)),
    Vector((-4.950318813323975, 11.35459041595459, 0.0)),
    Vector((2.100135326385498, 11.243268966674805, 0.0)),
    Vector((2.175142526626587, 4.452780246734619, 0.0)),
    Vector((0.7500491738319397, 4.452779769897461, 0.0)),
    Vector((0.6750447154045105, 0.667917013168335, 0.0)),
    Vector((0.0, 0.6679169535636902, 0.0)),
    Vector((0.0, 0.0, 8.7258939743042)),
    Vector((0.7500497698783875, 4.815185050688342e-08, 8.7258939743042)),
    Vector((0.7500501871109009, -3.450904130935669, 8.7258939743042)),
    Vector((0.0, -3.3395848274230957, 8.7258939743042)),
    Vector((0.0, -4.230140686035156, 8.7258939743042)),
    Vector((0.6750452518463135, -4.230140686035156, 8.7258939743042)),
    Vector((0.7500508427619934, -8.460281372070312, 8.7258939743042)),
    Vector((0.0, -8.571600914001465, 8.7258939743042)),
    Vector((0.0, -9.239518165588379, 8.7258939743042)),
    Vector((0.7500509023666382, -9.128198623657227, 8.7258939743042)),
    Vector((0.6750462651252747, -12.690422058105469, 8.7258939743042)),
    Vector((0.07500513643026352, -12.690422058105469, 8.7258939743042)),
    Vector((0.07500514388084412, -13.358339309692383, 8.7258939743042)),
    Vector((0.7500514984130859, -13.469657897949219, 8.7258939743042)),
    Vector((0.6750465035438538, -15.139451026916504, 8.7258939743042)),
    Vector((2.550175666809082, -15.028130531311035, 8.7258939743042)),
    Vector((2.475170135498047, -13.803616523742676, 8.7258939743042)),
    Vector((11.100762367248535, -13.803606033325195, 8.7258939743042)),
    Vector((11.100753784179688, -9.350826263427734, 8.7258939743042)),
    Vector((12.150825500488281, -9.23950481414795, 8.7258939743042)),
    Vector((12.825871467590332, -9.239503860473633, 8.7258939743042)),
    Vector((12.900873184204102, -7.792349815368652, 8.7258939743042)),
    Vector((13.350902557373047, -7.235751628875732, 8.7258939743042)),
    Vector((13.72592544555664, -6.456514358520508, 8.7258939743042)),
    Vector((13.950936317443848, -4.786721229553223, 8.7258939743042)),
    Vector((13.125879287719727, -4.007486820220947, 8.7258939743042)),
    Vector((13.200883865356445, -3.7848477363586426, 8.7258939743042)),
    Vector((12.900863647460938, -3.78484845161438, 8.7258939743042)),
    Vector((12.82585620880127, -2.5603342056274414, 8.7258939743042)),
    Vector((11.025735855102539, -2.3376989364624023, 8.7258939743042)),
    Vector((10.950722694396973, 2.1150805950164795, 8.7258939743042)),
    Vector((9.375617980957031, 2.1150779724121094, 8.7258939743042)),
    Vector((9.525615692138672, 9.573484420776367, 8.7258939743042)),
    Vector((10.800698280334473, 9.573486328125, 8.7258939743042)),
    Vector((10.57567024230957, 17.031890869140625, 8.7258939743042)),
    Vector((9.975628852844238, 19.369600296020508, 8.7258939743042)),
    Vector((9.000564575195312, 21.150711059570312, 8.7258939743042)),
    Vector((7.5004682540893555, 22.82050132751465, 8.7258939743042)),
    Vector((7.4254631996154785, 22.931819915771484, 8.7258939743042)),
    Vector((5.400335788726807, 24.045013427734375, 8.7258939743042)),
    Vector((3.3752095699310303, 24.824247360229492, 8.7258939743042)),
    Vector((1.0500651597976685, 25.158205032348633, 8.7258939743042)),
    Vector((-6.075376510620117, 25.380847930908203, 8.7258939743042)),
    Vector((-6.1503825187683105, 23.933692932128906, 8.7258939743042)),
    Vector((-13.200820922851562, 23.933706283569336, 8.7258939743042)),
    Vector((-13.125811576843262, 26.271413803100586, 8.7258939743042)),
    Vector((-15.075931549072266, 26.271419525146484, 8.7258939743042)),
    Vector((-15.07593059539795, 26.605377197265625, 8.7258939743042)),
    Vector((-15.375947952270508, 27.161975860595703, 8.7258939743042)),
    Vector((-16.050987243652344, 27.82989501953125, 8.7258939743042)),
    Vector((-16.726028442382812, 27.94121551513672, 8.7258939743042)),
    Vector((-17.851099014282227, 27.495941162109375, 8.7258939743042)),
    Vector((-18.376134872436523, 26.716707229614258, 8.7258939743042)),
    Vector((-18.451141357421875, 26.160110473632812, 8.7258939743042)),
    Vector((-18.901168823242188, 26.160110473632812, 8.7258939743042)),
    Vector((-18.976179122924805, 24.378999710083008, 8.7258939743042)),
    Vector((-24.376514434814453, 24.379018783569336, 8.7258939743042)),
    Vector((-24.45155143737793, 16.697975158691406, 8.7258939743042)),
    Vector((-25.126596450805664, 16.141380310058594, 8.7258939743042)),
    Vector((-22.12641716003418, 13.024422645568848, 8.7258939743042)),
    Vector((-18.526187896728516, 13.024410247802734, 8.7258939743042)),
    Vector((-18.45119285583496, 9.796144485473633, 8.7258939743042)),
    Vector((-13.65088176727295, 9.907450675964355, 8.7258939743042)),
    Vector((-13.650884628295898, 8.682936668395996, 8.7258939743042)),
    Vector((-12.900835990905762, 8.682934761047363, 8.7258939743042)),
    Vector((-12.975838661193848, 9.684810638427734, 8.7258939743042)),
    Vector((-9.675625801086426, 9.68480396270752, 8.7258939743042)),
    Vector((-9.675626754760742, 8.7942476272583, 8.7258939743042)),
    Vector((-8.850573539733887, 8.794246673583984, 8.7258939743042)),
    Vector((-8.700562477111816, 9.684802055358887, 8.7258939743042)),
    Vector((-5.025324821472168, 9.796117782592773, 8.7258939743042)),
    Vector((-4.950318813323975, 11.35459041595459, 8.7258939743042)),
    Vector((2.100135326385498, 11.243268966674805, 8.7258939743042)),
    Vector((2.175142526626587, 4.452780246734619, 8.7258939743042)),
    Vector((0.7500491738319397, 4.452779769897461, 8.7258939743042)),
    Vector((0.6750447154045105, 0.667917013168335, 8.7258939743042)),
    Vector((0.0, 0.6679169535636902, 8.7258939743042))
]
unitVectors = [
    Vector((0.9999999403953552, 6.419820408609667e-08, 0.0)),
    Vector((1.2090526979591232e-07, -1.0, 0.0)),
    Vector((-0.9891650080680847, 0.1468077152967453, 0.0)),
    Vector((0.0, -1.0, 0.0)),
    Vector((1.0, 0.0, 0.0)),
    Vector((0.017728440463542938, -0.9998427629470825, 0.0)),
    Vector((-0.9891649484634399, -0.14680790901184082, 0.0)),
    Vector((0.0, -1.0, 0.0)),
    Vector((0.9891650676727295, 0.14680790901184082, 0.0)),
    Vector((-0.02105090208351612, -0.9997783899307251, 0.0)),
    Vector((-1.0, 0.0, 0.0)),
    Vector((1.1154945411817607e-08, -1.0, 0.0)),
    Vector((0.9866743087768555, -0.1627076417207718, 0.0)),
    Vector((-0.04487348720431328, -0.9989926815032959, 0.0)),
    Vector((0.9982423782348633, 0.05926249548792839, 0.0)),
    Vector((-0.06113871559500694, 0.9981292486190796, 0.0)),
    Vector((1.0, 1.2161967788415495e-06, 0.0)),
    Vector((-1.927575340232579e-06, 1.0, 0.0)),
    Vector((0.994427502155304, 0.10542242974042892, 0.0)),
    Vector((1.0, 1.412754613738798e-06, 0.0)),
    Vector((0.0517575778067112, 0.998659610748291, 0.0)),
    Vector((0.6287338137626648, 0.7776206135749817, 0.0)),
    Vector((0.4336603581905365, 0.9010764956474304, 0.0)),
    Vector((0.13354668021202087, 0.99104243516922, 0.0)),
    Vector((-0.7270070314407349, 0.6866299510002136, 0.0)),
    Vector((0.3192584812641144, 0.94766765832901, 0.0)),
    Vector((-1.0, -2.3840250378270866e-06, 0.0)),
    Vector((-0.06114025413990021, 0.9981291890144348, 0.0)),
    Vector((-0.9924384951591492, 0.12274280190467834, 0.0)),
    Vector((-0.016843976452946663, 0.9998581409454346, 0.0)),
    Vector((-0.9999999403953552, -1.6650349152769195e-06, 0.0)),
    Vector((0.02010716125369072, 0.9997978210449219, 0.0)),
    Vector((0.9999999403953552, 1.4958627616579179e-06, 0.0)),
    Vector((-0.03015734627842903, 0.999545156955719, 0.0)),
    Vector((-0.24861976504325867, 0.9686011075973511, 0.0)),
    Vector((-0.4801987111568451, 0.8771597146987915, 0.0)),
    Vector((-0.668296217918396, 0.7438952922821045, 0.0)),
    Vector((-0.5587818026542664, 0.8293147683143616, 0.0)),
    Vector((-0.8763304352760315, 0.4817105829715729, 0.0)),
    Vector((-0.9332932233810425, 0.35911527276039124, 0.0)),
    Vector((-0.9898422956466675, 0.14216984808444977, 0.0)),
    Vector((-0.9995121955871582, 0.031230947002768517, 0.0)),
    Vector((-0.05176049843430519, -0.9986594915390015, 0.0)),
    Vector((-1.0, 1.8937035974886385e-06, 0.0)),
    Vector((0.032070208340883255, 0.9994856119155884, 0.0)),
    Vector((-1.0, 2.9342018024181016e-06, 0.0)),
    Vector((2.8556742108776234e-06, 1.0, 0.0)),
    Vector((-0.4744803011417389, 0.8802660703659058, 0.0)),
    Vector((-0.7108457088470459, 0.70334792137146, 0.0)),
    Vector((-0.9866736531257629, 0.16271154582500458, 0.0)),
    Vector((-0.9298253059387207, -0.36800122261047363, 0.0)),
    Vector((-0.5587802529335022, -0.8293157815933228, 0.0)),
    Vector((-0.13355191051959991, -0.9910417795181274, 0.0)),
    Vector((-1.0, 0.0, 0.0)),
    Vector((-0.04207703843712807, -0.9991143941879272, 0.0)),
    Vector((-1.0, 3.531907850629068e-06, 0.0)),
    Vector((-0.009768649004399776, -0.9999522566795349, 0.0)),
    Vector((-0.771551251411438, -0.6361671686172485, 0.0)),
    Vector((0.6934815049171448, -0.7204744219779968, 0.0)),
    Vector((1.0, -3.4436045552865835e-06, 0.0)),
    Vector((0.0232244860380888, -0.9997302293777466, 0.0)),
    Vector((0.9997312426567078, 0.02318105474114418, 0.0)),
    Vector((-2.3364559638139326e-06, -1.0, 0.0)),
    Vector((0.9999999403953552, -2.54296651291952e-06, 0.0)),
    Vector((-0.07465333491563797, 0.9972094893455505, 0.0)),
    Vector((1.0, -2.02281512429181e-06, 0.0)),
    Vector((-1.0708747595344903e-06, -1.0, 0.0)),
    Vector((1.0, -1.1558943242562236e-06, 0.0)),
    Vector((0.16610653698444366, 0.9861077666282654, 0.0)),
    Vector((0.9995415806770325, 0.030274150893092155, 0.0)),
    Vector((0.04807225614786148, 0.9988439083099365, 0.0)),
    Vector((0.9998753070831299, -0.015787290409207344, 0.0)),
    Vector((0.011045247316360474, -0.9999390244483948, 0.0)),
    Vector((-1.0, -3.3460062809353985e-07, 0.0)),
    Vector((-0.019813066348433495, -0.9998037219047546, 0.0)),
    Vector((-1.0, -8.82973267835041e-08, 0.0)),
    Vector((0.0, -1.0, 0.0))
]
holesInfo = None
firstVertIndex = 77
numPolygonVerts = 77
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
