import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((3.950855016708374, -0.9462144374847412, 0.0)),
    Vector((4.1085734367370605, -0.28942936658859253, 0.0)),
    Vector((7.207747459411621, -1.0352672338485718, 0.0)),
    Vector((8.840127944946289, 5.744091987609863, 0.0)),
    Vector((5.149511337280273, 6.6346435546875, 0.0)),
    Vector((5.307229518890381, 7.313692569732666, 0.0)),
    Vector((2.7048730850219727, 7.937080383300781, 0.0)),
    Vector((2.799504041671753, 8.326698303222656, 0.0)),
    Vector((-2.2474887371063232, 9.540081024169922, 0.0)),
    Vector((-2.3421199321746826, 9.16159439086914, 0.0)),
    Vector((-5.551691055297852, 9.94083309173584, 0.0)),
    Vector((-5.701524257659912, 9.339707374572754, 0.0)),
    Vector((-8.753376007080078, 10.074419975280762, 0.0)),
    Vector((-10.448862075805664, 3.0390305519104004, 0.0)),
    Vector((-7.4285502433776855, 2.3043177127838135, 0.0)),
    Vector((-7.546839714050293, 1.8256441354751587, 0.0)),
    Vector((-3.651188373565674, 0.8348972201347351, 0.0)),
    Vector((-3.5250136852264404, 0.4341469705104828, 0.0)),
    Vector((-3.319979429244995, 0.05566060543060303, 0.0)),
    Vector((-3.036085844039917, -0.26716604828834534, 0.0)),
    Vector((-2.696990728378296, -0.523201048374176, 0.0)),
    Vector((-2.310579776763916, -0.7013123631477356, 0.0)),
    Vector((-1.892625093460083, -0.8015000820159912, 0.0)),
    Vector((-1.4667844772338867, -0.801500141620636, 0.0)),
    Vector((-1.0488297939300537, -0.7235766053199768, 0.0)),
    Vector((-0.6545328497886658, -0.5565974116325378, 0.0)),
    Vector((-0.299665629863739, -0.31169456243515015, 0.0)),
    Vector((0.0, 0.0, 9.192001342773438)),
    Vector((3.950855016708374, -0.9462144374847412, 9.192001342773438)),
    Vector((4.1085734367370605, -0.28942936658859253, 9.192001342773438)),
    Vector((7.207747459411621, -1.0352672338485718, 9.192001342773438)),
    Vector((8.840127944946289, 5.744091987609863, 9.192001342773438)),
    Vector((5.149511337280273, 6.6346435546875, 9.192001342773438)),
    Vector((5.307229518890381, 7.313692569732666, 9.192001342773438)),
    Vector((2.7048730850219727, 7.937080383300781, 9.192001342773438)),
    Vector((2.799504041671753, 8.326698303222656, 9.192001342773438)),
    Vector((-2.2474887371063232, 9.540081024169922, 9.192001342773438)),
    Vector((-2.3421199321746826, 9.16159439086914, 9.192001342773438)),
    Vector((-5.551691055297852, 9.94083309173584, 9.192001342773438)),
    Vector((-5.701524257659912, 9.339707374572754, 9.192001342773438)),
    Vector((-8.753376007080078, 10.074419975280762, 9.192001342773438)),
    Vector((-10.448862075805664, 3.0390305519104004, 9.192001342773438)),
    Vector((-7.4285502433776855, 2.3043177127838135, 9.192001342773438)),
    Vector((-7.546839714050293, 1.8256441354751587, 9.192001342773438)),
    Vector((-3.651188373565674, 0.8348972201347351, 9.192001342773438)),
    Vector((-3.5250136852264404, 0.4341469705104828, 9.192001342773438)),
    Vector((-3.319979429244995, 0.05566060543060303, 9.192001342773438)),
    Vector((-3.036085844039917, -0.26716604828834534, 9.192001342773438)),
    Vector((-2.696990728378296, -0.523201048374176, 9.192001342773438)),
    Vector((-2.310579776763916, -0.7013123631477356, 9.192001342773438)),
    Vector((-1.892625093460083, -0.8015000820159912, 9.192001342773438)),
    Vector((-1.4667844772338867, -0.801500141620636, 9.192001342773438)),
    Vector((-1.0488297939300537, -0.7235766053199768, 9.192001342773438)),
    Vector((-0.6545328497886658, -0.5565974116325378, 9.192001342773438)),
    Vector((-0.299665629863739, -0.31169456243515015, 9.192001342773438))
]
unitVectors = [
    Vector((0.9724984169006348, -0.23290960490703583, 0.0)),
    Vector((0.2334989458322525, 0.9723570346832275, 0.0)),
    Vector((0.9722421765327454, -0.23397687077522278, 0.0)),
    Vector((0.2340961992740631, 0.9722134470939636, 0.0)),
    Vector((-0.9720993638038635, 0.23456910252571106, 0.0)),
    Vector((0.2262410670518875, 0.9740714430809021, 0.0)),
    Vector((-0.9724870324134827, 0.23295676708221436, 0.0)),
    Vector((0.23601962625980377, 0.9717482924461365, 0.0)),
    Vector((-0.9722952842712402, 0.23375628888607025, 0.0)),
    Vector((-0.24255862832069397, -0.9701367616653442, 0.0)),
    Vector((-0.9717695713043213, 0.23593197762966156, 0.0)),
    Vector((-0.24185462296009064, -0.9703125357627869, 0.0)),
    Vector((-0.9722231030464172, 0.23405611515045166, 0.0)),
    Vector((-0.23428645730018616, -0.9721675515174866, 0.0)),
    Vector((0.9716644287109375, -0.2363644391298294, 0.0)),
    Vector((-0.23990264534950256, -0.9707970023155212, 0.0)),
    Vector((0.969149112701416, -0.2464752048254013, 0.0)),
    Vector((0.3003130853176117, -0.9538406729698181, 0.0)),
    Vector((0.4763205349445343, -0.8792717456817627, 0.0)),
    Vector((0.6603736877441406, -0.7509371042251587, 0.0)),
    Vector((0.7980599403381348, -0.6025780439376831, 0.0)),
    Vector((0.9081669449806213, -0.41860824823379517, 0.0)),
    Vector((0.972451388835907, -0.2331058531999588, 0.0)),
    Vector((1.0, -1.3996937298088596e-07, 0.0)),
    Vector((0.9830604195594788, 0.18328192830085754, 0.0)),
    Vector((0.920832097530365, 0.3899593949317932, 0.0)),
    Vector((0.8230319023132324, 0.5679951906204224, 0.0)),
    Vector((0.6930598020553589, 0.7208800315856934, 0.0))
]
holesInfo = None
firstVertIndex = 28
numPolygonVerts = 28
faces = []


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