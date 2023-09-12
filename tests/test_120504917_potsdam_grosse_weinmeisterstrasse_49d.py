import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((-0.5771195888519287, -1.0130072832107544, 0.0)),
    Vector((-0.936970591545105, -0.8015002608299255, 0.0)),
    Vector((-2.6072239875793457, -3.7069382667541504, 0.0)),
    Vector((-3.6935670375823975, -3.0835485458374023, 0.0)),
    Vector((-7.611202716827393, -9.929692268371582, 0.0)),
    Vector((-9.261087417602539, -8.983473777770996, 0.0)),
    Vector((-7.13592004776001, -5.276538848876953, 0.0)),
    Vector((-14.203932762145996, -1.2244938611984253, 0.0)),
    Vector((-20.002328872680664, -11.343415260314941, 0.0)),
    Vector((-3.9923255443573, -20.505048751831055, 0.0)),
    Vector((-0.3259037137031555, -14.104179382324219, 0.0)),
    Vector((-1.2153490781784058, -13.592109680175781, 0.0)),
    Vector((4.94965124130249, -2.827512502670288, 0.0)),
    Vector((6.585956573486328, -3.76259446144104, 0.0)),
    Vector((12.526870727539062, 6.612393856048584, 0.0)),
    Vector((3.924403190612793, 11.532700538635254, 0.0)),
    Vector((-2.0165228843688965, 1.157723069190979, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((-0.5771195888519287, -1.0130072832107544, 2.99674654006958)),
    Vector((-0.936970591545105, -0.8015002608299255, 2.99674654006958)),
    Vector((-2.6072239875793457, -3.7069382667541504, 2.99674654006958)),
    Vector((-3.6935670375823975, -3.0835485458374023, 2.99674654006958)),
    Vector((-7.611202716827393, -9.929692268371582, 2.99674654006958)),
    Vector((-9.261087417602539, -8.983473777770996, 2.99674654006958)),
    Vector((-7.13592004776001, -5.276538848876953, 2.99674654006958)),
    Vector((-14.203932762145996, -1.2244938611984253, 2.99674654006958)),
    Vector((-20.002328872680664, -11.343415260314941, 2.99674654006958)),
    Vector((-3.9923255443573, -20.505048751831055, 2.99674654006958)),
    Vector((-0.3259037137031555, -14.104179382324219, 2.99674654006958)),
    Vector((-1.2153490781784058, -13.592109680175781, 2.99674654006958)),
    Vector((4.94965124130249, -2.827512502670288, 2.99674654006958)),
    Vector((6.585956573486328, -3.76259446144104, 2.99674654006958)),
    Vector((12.526870727539062, 6.612393856048584, 2.99674654006958)),
    Vector((3.924403190612793, 11.532700538635254, 2.99674654006958)),
    Vector((-2.0165228843688965, 1.157723069190979, 2.99674654006958)),
    Vector((0.0, 0.0, 2.99674654006958)),
]
unitVectors = [
    Vector((-0.8621122241020203, 0.5067174434661865, 0.0)),
    Vector((-0.4983873963356018, -0.866954505443573, 0.0)),
    Vector((-0.8673397898674011, 0.49771633744239807, 0.0)),
    Vector((-0.49666962027549744, -0.8679397106170654, 0.0)),
    Vector((-0.8674657940864563, 0.49749669432640076, 0.0)),
    Vector((0.4973590672016144, 0.8675447106361389, 0.0)),
    Vector((-0.8675454258918762, 0.4973580539226532, 0.0)),
    Vector((-0.4971828758716583, -0.8676458597183228, 0.0)),
    Vector((0.8679380416870117, -0.4966726005077362, 0.0)),
    Vector((0.4970361888408661, 0.8677298426628113, 0.0)),
    Vector((-0.8666372299194336, 0.498938649892807, 0.0)),
    Vector((0.49697744846343994, 0.8677634000778198, 0.0)),
    Vector((0.8682316541671753, -0.49615907669067383, 0.0)),
    Vector((0.4969174265861511, 0.8677978515625, 0.0)),
    Vector((-0.8680427074432373, 0.49648967385292053, 0.0)),
    Vector((-0.49691858887672424, -0.8677971959114075, 0.0)),
    Vector((0.8672364950180054, -0.4978964924812317, 0.0)),
    Vector((-0.4950123131275177, -0.8688859343528748, 0.0)),
]
holesInfo = None
firstVertIndex = 18
numPolygonVerts = 18
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
