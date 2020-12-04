import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-2.553337335586548, -4.0742926597595215, 0.0)),
    Vector((-3.113825798034668, -3.7292020320892334, 0.0)),
    Vector((-4.276322364807129, -5.565972805023193, 0.0)),
    Vector((-3.708914041519165, -5.9221954345703125, 0.0)),
    Vector((-4.255564212799072, -6.768223285675049, 0.0)),
    Vector((-3.812709093093872, -7.035390377044678, 0.0)),
    Vector((-7.646186351776123, -13.235881805419922, 0.0)),
    Vector((-9.362252235412598, -12.211739540100098, 0.0)),
    Vector((-13.0850248336792, -18.14505958557129, 0.0)),
    Vector((-6.0339250564575195, -22.631248474121094, 0.0)),
    Vector((-2.4910671710968018, -17.05414581298828, 0.0)),
    Vector((-1.2178552150726318, -17.866777420043945, 0.0)),
    Vector((0.03459814563393593, -15.851895332336426, 0.0)),
    Vector((0.7957574725151062, -16.330568313598633, 0.0)),
    Vector((1.7921836376190186, -14.749832153320312, 0.0)),
    Vector((1.0241048336029053, -14.260026931762695, 0.0)),
    Vector((1.591513991355896, -13.347207069396973, 0.0)),
    Vector((2.3665122985839844, -13.83701229095459, 0.0)),
    Vector((3.3698570728302, -12.234010696411133, 0.0)),
    Vector((2.601778507232666, -11.755337715148926, 0.0)),
    Vector((2.8647239208221436, -11.321191787719727, 0.0)),
    Vector((5.452664852142334, -12.924189567565918, 0.0)),
    Vector((6.4490885734558105, -11.321187973022461, 0.0)),
    Vector((7.355560302734375, -11.888916015625, 0.0)),
    Vector((9.618270874023438, -8.293292999267578, 0.0)),
    Vector((8.704879760742188, -7.714433193206787, 0.0)),
    Vector((9.708221435546875, -6.100298881530762, 0.0)),
    Vector((5.418056488037109, -3.3952414989471436, 0.0)),
    Vector((5.694840431213379, -2.9610953330993652, 0.0)),
    Vector((4.760692119598389, -2.382234811782837, 0.0)),
    Vector((4.490828037261963, -2.8163812160491943, 0.0)),
    Vector((2.1589181423187256, -1.3580973148345947, 0.0)),
    Vector((2.4287827014923096, -0.9128192663192749, 0.0)),
    Vector((1.508474588394165, -0.333958238363266, 0.0)),
    Vector((1.2316904067993164, -0.7681043148040771, 0.0)),
    Vector((0.0, 0.0, 3.1747679710388184)),
    Vector((-2.553337335586548, -4.0742926597595215, 3.1747679710388184)),
    Vector((-3.113825798034668, -3.7292020320892334, 3.1747679710388184)),
    Vector((-4.276322364807129, -5.565972805023193, 3.1747679710388184)),
    Vector((-3.708914041519165, -5.9221954345703125, 3.1747679710388184)),
    Vector((-4.255564212799072, -6.768223285675049, 3.1747679710388184)),
    Vector((-3.812709093093872, -7.035390377044678, 3.1747679710388184)),
    Vector((-7.646186351776123, -13.235881805419922, 3.1747679710388184)),
    Vector((-9.362252235412598, -12.211739540100098, 3.1747679710388184)),
    Vector((-13.0850248336792, -18.14505958557129, 3.1747679710388184)),
    Vector((-6.0339250564575195, -22.631248474121094, 3.1747679710388184)),
    Vector((-2.4910671710968018, -17.05414581298828, 3.1747679710388184)),
    Vector((-1.2178552150726318, -17.866777420043945, 3.1747679710388184)),
    Vector((0.03459814563393593, -15.851895332336426, 3.1747679710388184)),
    Vector((0.7957574725151062, -16.330568313598633, 3.1747679710388184)),
    Vector((1.7921836376190186, -14.749832153320312, 3.1747679710388184)),
    Vector((1.0241048336029053, -14.260026931762695, 3.1747679710388184)),
    Vector((1.591513991355896, -13.347207069396973, 3.1747679710388184)),
    Vector((2.3665122985839844, -13.83701229095459, 3.1747679710388184)),
    Vector((3.3698570728302, -12.234010696411133, 3.1747679710388184)),
    Vector((2.601778507232666, -11.755337715148926, 3.1747679710388184)),
    Vector((2.8647239208221436, -11.321191787719727, 3.1747679710388184)),
    Vector((5.452664852142334, -12.924189567565918, 3.1747679710388184)),
    Vector((6.4490885734558105, -11.321187973022461, 3.1747679710388184)),
    Vector((7.355560302734375, -11.888916015625, 3.1747679710388184)),
    Vector((9.618270874023438, -8.293292999267578, 3.1747679710388184)),
    Vector((8.704879760742188, -7.714433193206787, 3.1747679710388184)),
    Vector((9.708221435546875, -6.100298881530762, 3.1747679710388184)),
    Vector((5.418056488037109, -3.3952414989471436, 3.1747679710388184)),
    Vector((5.694840431213379, -2.9610953330993652, 3.1747679710388184)),
    Vector((4.760692119598389, -2.382234811782837, 3.1747679710388184)),
    Vector((4.490828037261963, -2.8163812160491943, 3.1747679710388184)),
    Vector((2.1589181423187256, -1.3580973148345947, 3.1747679710388184)),
    Vector((2.4287827014923096, -0.9128192663192749, 3.1747679710388184)),
    Vector((1.508474588394165, -0.333958238363266, 3.1747679710388184)),
    Vector((1.2316904067993164, -0.7681043148040771, 3.1747679710388184))
]
unitVectors = [
    Vector((-0.5310311317443848, -0.8473522663116455, 0.0)),
    Vector((-0.8515398502349854, 0.5242898464202881, 0.0)),
    Vector((-0.5347921252250671, -0.8449835777282715, 0.0)),
    Vector((0.8469282984733582, -0.5317071080207825, 0.0)),
    Vector((-0.5427055954933167, -0.8399230241775513, 0.0)),
    Vector((0.8562501668930054, -0.5165613889694214, 0.0)),
    Vector((-0.5258663296699524, -0.8505671620368958, 0.0)),
    Vector((-0.858704149723053, 0.5124716758728027, 0.0)),
    Vector((-0.5314813256263733, -0.8470699787139893, 0.0)),
    Vector((0.8437088131904602, -0.5368009805679321, 0.0)),
    Vector((0.5362066030502319, 0.8440867066383362, 0.0)),
    Vector((0.8429393172264099, -0.5380087494850159, 0.0)),
    Vector((0.5279216766357422, 0.849293053150177, 0.0)),
    Vector((0.846521258354187, -0.5323548913002014, 0.0)),
    Vector((0.5332530736923218, 0.8459557294845581, 0.0)),
    Vector((-0.8431500196456909, 0.5376783013343811, 0.0)),
    Vector((0.5279210805892944, 0.8492934107780457, 0.0)),
    Vector((0.8453251123428345, -0.5342523455619812, 0.0)),
    Vector((0.5305573344230652, 0.8476490378379822, 0.0)),
    Vector((-0.8486809730529785, 0.5289050936698914, 0.0)),
    Vector((0.5180519223213196, 0.8553491830825806, 0.0)),
    Vector((0.8501270413398743, -0.5265775918960571, 0.0)),
    Vector((0.5279200673103333, 0.8492940068244934, 0.0)),
    Vector((0.847500741481781, -0.5307942032814026, 0.0)),
    Vector((0.5326109528541565, 0.8463601469993591, 0.0)),
    Vector((-0.8446606993675232, 0.5353020429611206, 0.0)),
    Vector((0.5279192924499512, 0.8492945432662964, 0.0)),
    Vector((-0.845890998840332, 0.5333556532859802, 0.0)),
    Vector((0.5375790596008301, 0.8432132601737976, 0.0)),
    Vector((-0.8500295281410217, 0.5267349481582642, 0.0)),
    Vector((-0.5279189944267273, -0.8492946624755859, 0.0)),
    Vector((-0.8478609323501587, 0.5302186012268066, 0.0)),
    Vector((0.5183004140853882, 0.8551986217498779, 0.0)),
    Vector((-0.8464783430099487, 0.5324230790138245, 0.0)),
    Vector((-0.5375794768333435, -0.8432130217552185, 0.0)),
    Vector((-0.8485249280929565, 0.529155433177948, 0.0))
]
holesInfo = None
firstVertIndex = 36
numPolygonVerts = 36
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