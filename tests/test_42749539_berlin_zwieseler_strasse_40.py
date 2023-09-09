import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-4.622979164123535, -53.099395751953125, 0.0)),
    Vector((-8.20883846282959, -52.84335708618164, 0.0)),
    Vector((-8.235952377319336, -53.043731689453125, 0.0)),
    Vector((-8.56810188293457, -53.032596588134766, 0.0)),
    Vector((-8.527430534362793, -52.82109069824219, 0.0)),
    Vector((-9.564550399780273, -52.7431640625, 0.0)),
    Vector((-9.673009872436523, -53.94541549682617, 0.0)),
    Vector((-9.245960235595703, -54.023338317871094, 0.0)),
    Vector((-9.449321746826172, -56.48350143432617, 0.0)),
    Vector((-9.883150100708008, -56.43897247314453, 0.0)),
    Vector((-9.984830856323242, -57.59669494628906, 0.0)),
    Vector((-9.544224739074707, -57.66348648071289, 0.0)),
    Vector((-9.74080753326416, -60.07912063598633, 0.0)),
    Vector((-10.194972038269043, -60.03459167480469, 0.0)),
    Vector((-10.296652793884277, -61.2257080078125, 0.0)),
    Vector((-9.842488288879395, -61.292503356933594, 0.0)),
    Vector((-9.876381874084473, -61.71551513671875, 0.0)),
    Vector((-9.578125, -61.71551513671875, 0.0)),
    Vector((-9.815381050109863, -64.45397186279297, 0.0)),
    Vector((-10.093301773071289, -64.42057800292969, 0.0)),
    Vector((-10.127195358276367, -64.71000671386719, 0.0)),
    Vector((-10.594917297363281, -64.63208770751953, 0.0)),
    Vector((-10.703377723693848, -65.85659790039062, 0.0)),
    Vector((-10.222098350524902, -65.86772918701172, 0.0)),
    Vector((-10.425460815429688, -68.35015869140625, 0.0)),
    Vector((-10.913518905639648, -68.31675720214844, 0.0)),
    Vector((-11.02875804901123, -69.59693145751953, 0.0)),
    Vector((-10.533921241760254, -69.67485809326172, 0.0)),
    Vector((-10.723726272583008, -71.93464660644531, 0.0)),
    Vector((-11.225341796875, -71.9012451171875, 0.0)),
    Vector((-11.347359657287598, -73.27047729492188, 0.0)),
    Vector((-10.14077091217041, -73.38179779052734, 0.0)),
    Vector((-10.14077091217041, -73.25934600830078, 0.0)),
    Vector((-9.862848281860352, -73.25934600830078, 0.0)),
    Vector((-9.89674186706543, -73.40406036376953, 0.0)),
    Vector((-6.310867786407471, -73.73802947998047, 0.0)),
    Vector((-11.062780380249023, -126.94873809814453, 0.0)),
    Vector((15.048641204833984, -129.13058471679688, 0.0)),
    Vector((15.923048973083496, -116.58487701416016, 0.0)),
    Vector((7.9106831550598145, -115.75, 0.0)),
    Vector((8.080145835876465, -113.81304168701172, 0.0)),
    Vector((9.61889934539795, -113.92435455322266, 0.0)),
    Vector((9.829032897949219, -111.73136138916016, 0.0)),
    Vector((8.269944190979004, -111.6423110961914, 0.0)),
    Vector((11.435471534729004, -69.1405258178711, 0.0)),
    Vector((12.465816497802734, -69.25183868408203, 0.0)),
    Vector((13.13011646270752, -68.77316284179688, 0.0)),
    Vector((13.469037055969238, -65.4892349243164, 0.0)),
    Vector((13.726622581481934, -65.52263641357422, 0.0)),
    Vector((13.74695873260498, -65.45584106445312, 0.0)),
    Vector((13.868972778320312, -65.45584106445312, 0.0)),
    Vector((13.902865409851074, -65.17754364013672, 0.0)),
    Vector((13.760515213012695, -65.14414978027344, 0.0)),
    Vector((13.740179061889648, -65.08848571777344, 0.0)),
    Vector((13.5097074508667, -65.08848571777344, 0.0)),
    Vector((13.841848373413086, -61.82682418823242, 0.0)),
    Vector((13.286004066467285, -61.27022933959961, 0.0)),
    Vector((12.533581733703613, -61.1700439453125, 0.0)),
    Vector((16.77680778503418, -14.059622764587402, 0.0)),
    Vector((25.073705673217773, -14.816559791564941, 0.0)),
    Vector((26.042966842651367, -2.4266955852508545, 0.0)),
    Vector((0.0, 0.0, 9.117474555969238)),
    Vector((-4.622979164123535, -53.099395751953125, 9.117474555969238)),
    Vector((-8.20883846282959, -52.84335708618164, 9.117474555969238)),
    Vector((-8.235952377319336, -53.043731689453125, 9.117474555969238)),
    Vector((-8.56810188293457, -53.032596588134766, 9.117474555969238)),
    Vector((-8.527430534362793, -52.82109069824219, 9.117474555969238)),
    Vector((-9.564550399780273, -52.7431640625, 9.117474555969238)),
    Vector((-9.673009872436523, -53.94541549682617, 9.117474555969238)),
    Vector((-9.245960235595703, -54.023338317871094, 9.117474555969238)),
    Vector((-9.449321746826172, -56.48350143432617, 9.117474555969238)),
    Vector((-9.883150100708008, -56.43897247314453, 9.117474555969238)),
    Vector((-9.984830856323242, -57.59669494628906, 9.117474555969238)),
    Vector((-9.544224739074707, -57.66348648071289, 9.117474555969238)),
    Vector((-9.74080753326416, -60.07912063598633, 9.117474555969238)),
    Vector((-10.194972038269043, -60.03459167480469, 9.117474555969238)),
    Vector((-10.296652793884277, -61.2257080078125, 9.117474555969238)),
    Vector((-9.842488288879395, -61.292503356933594, 9.117474555969238)),
    Vector((-9.876381874084473, -61.71551513671875, 9.117474555969238)),
    Vector((-9.578125, -61.71551513671875, 9.117474555969238)),
    Vector((-9.815381050109863, -64.45397186279297, 9.117474555969238)),
    Vector((-10.093301773071289, -64.42057800292969, 9.117474555969238)),
    Vector((-10.127195358276367, -64.71000671386719, 9.117474555969238)),
    Vector((-10.594917297363281, -64.63208770751953, 9.117474555969238)),
    Vector((-10.703377723693848, -65.85659790039062, 9.117474555969238)),
    Vector((-10.222098350524902, -65.86772918701172, 9.117474555969238)),
    Vector((-10.425460815429688, -68.35015869140625, 9.117474555969238)),
    Vector((-10.913518905639648, -68.31675720214844, 9.117474555969238)),
    Vector((-11.02875804901123, -69.59693145751953, 9.117474555969238)),
    Vector((-10.533921241760254, -69.67485809326172, 9.117474555969238)),
    Vector((-10.723726272583008, -71.93464660644531, 9.117474555969238)),
    Vector((-11.225341796875, -71.9012451171875, 9.117474555969238)),
    Vector((-11.347359657287598, -73.27047729492188, 9.117474555969238)),
    Vector((-10.14077091217041, -73.38179779052734, 9.117474555969238)),
    Vector((-10.14077091217041, -73.25934600830078, 9.117474555969238)),
    Vector((-9.862848281860352, -73.25934600830078, 9.117474555969238)),
    Vector((-9.89674186706543, -73.40406036376953, 9.117474555969238)),
    Vector((-6.310867786407471, -73.73802947998047, 9.117474555969238)),
    Vector((-11.062780380249023, -126.94873809814453, 9.117474555969238)),
    Vector((15.048641204833984, -129.13058471679688, 9.117474555969238)),
    Vector((15.923048973083496, -116.58487701416016, 9.117474555969238)),
    Vector((7.9106831550598145, -115.75, 9.117474555969238)),
    Vector((8.080145835876465, -113.81304168701172, 9.117474555969238)),
    Vector((9.61889934539795, -113.92435455322266, 9.117474555969238)),
    Vector((9.829032897949219, -111.73136138916016, 9.117474555969238)),
    Vector((8.269944190979004, -111.6423110961914, 9.117474555969238)),
    Vector((11.435471534729004, -69.1405258178711, 9.117474555969238)),
    Vector((12.465816497802734, -69.25183868408203, 9.117474555969238)),
    Vector((13.13011646270752, -68.77316284179688, 9.117474555969238)),
    Vector((13.469037055969238, -65.4892349243164, 9.117474555969238)),
    Vector((13.726622581481934, -65.52263641357422, 9.117474555969238)),
    Vector((13.74695873260498, -65.45584106445312, 9.117474555969238)),
    Vector((13.868972778320312, -65.45584106445312, 9.117474555969238)),
    Vector((13.902865409851074, -65.17754364013672, 9.117474555969238)),
    Vector((13.760515213012695, -65.14414978027344, 9.117474555969238)),
    Vector((13.740179061889648, -65.08848571777344, 9.117474555969238)),
    Vector((13.5097074508667, -65.08848571777344, 9.117474555969238)),
    Vector((13.841848373413086, -61.82682418823242, 9.117474555969238)),
    Vector((13.286004066467285, -61.27022933959961, 9.117474555969238)),
    Vector((12.533581733703613, -61.1700439453125, 9.117474555969238)),
    Vector((16.77680778503418, -14.059622764587402, 9.117474555969238)),
    Vector((25.073705673217773, -14.816559791564941, 9.117474555969238)),
    Vector((26.042966842651367, -2.4266955852508545, 9.117474555969238))
]
unitVectors = [
    Vector((-0.08673463761806488, -0.9962313771247864, 0.0)),
    Vector((-0.9974606037139893, 0.0712210014462471, 0.0)),
    Vector((-0.1340940296649933, -0.990968644618988, 0.0)),
    Vector((-0.9994385242462158, 0.03350554034113884, 0.0)),
    Vector((0.18883457779884338, 0.9820088744163513, 0.0)),
    Vector((-0.9971890449523926, 0.07492633163928986, 0.0)),
    Vector((-0.08984875679016113, -0.9959554076194763, 0.0)),
    Vector((0.9837572574615479, -0.17950405180454254, 0.0)),
    Vector((-0.08238082379102707, -0.9966009259223938, 0.0)),
    Vector((-0.9947735667228699, 0.10210543870925903, 0.0)),
    Vector((-0.0874914675951004, -0.9961652755737305, 0.0)),
    Vector((0.9887045621871948, -0.14987784624099731, 0.0)),
    Vector((-0.08111122995615005, -0.9967049956321716, 0.0)),
    Vector((-0.9952278733253479, 0.09757799655199051, 0.0)),
    Vector((-0.08505657315254211, -0.996376097202301, 0.0)),
    Vector((0.9893571734428406, -0.14550775289535522, 0.0)),
    Vector((-0.07986848801374435, -0.9968054294586182, 0.0)),
    Vector((1.0, 0.0, 0.0)),
    Vector((-0.08631525188684464, -0.9962679147720337, 0.0)),
    Vector((-0.9928584694862366, 0.11929796636104584, 0.0)),
    Vector((-0.11631032079458237, -0.993212878704071, 0.0)),
    Vector((-0.9864057898521423, 0.16432788968086243, 0.0)),
    Vector((-0.08822911977767944, -0.9961001873016357, 0.0)),
    Vector((0.9997326731681824, -0.023122351616621017, 0.0)),
    Vector((-0.08164723217487335, -0.9966613054275513, 0.0)),
    Vector((-0.9976662993431091, 0.06827782094478607, 0.0)),
    Vector((-0.089655801653862, -0.995972752571106, 0.0)),
    Vector((0.9878261089324951, -0.15556232631206512, 0.0)),
    Vector((-0.08369767665863037, -0.9964911937713623, 0.0)),
    Vector((-0.9977903366088867, 0.06644069403409958, 0.0)),
    Vector((-0.0887623205780983, -0.9960528016090393, 0.0)),
    Vector((0.9957709908485413, -0.09187033772468567, 0.0)),
    Vector((0.0, 1.0, 0.0)),
    Vector((1.0, 0.0, 0.0)),
    Vector((-0.22803926467895508, -0.9736519455909729, 0.0)),
    Vector((0.9956910014152527, -0.09273333102464676, 0.0)),
    Vector((-0.08894970268011093, -0.9960361123085022, 0.0)),
    Vector((0.9965271353721619, -0.0832688957452774, 0.0)),
    Vector((0.06952908635139465, 0.9975799322128296, 0.0)),
    Vector((-0.9946151971817017, 0.10363747924566269, 0.0)),
    Vector((0.08715613931417465, 0.99619460105896, 0.0)),
    Vector((0.9973936676979065, -0.07215109467506409, 0.0)),
    Vector((0.09538354724645615, 0.9954406023025513, 0.0)),
    Vector((-0.9983728528022766, 0.057023949921131134, 0.0)),
    Vector((0.07427413761615753, 0.9972378015518188, 0.0)),
    Vector((0.9942148327827454, -0.10740955919027328, 0.0)),
    Vector((0.8113142848014832, 0.5846102237701416, 0.0)),
    Vector((0.10266056656837463, 0.9947165250778198, 0.0)),
    Vector((0.9916971921920776, -0.12859481573104858, 0.0)),
    Vector((0.29125508666038513, 0.9566453695297241, 0.0)),
    Vector((1.0, 0.0, 0.0)),
    Vector((0.12089242786169052, 0.9926656484603882, 0.0)),
    Vector((-0.9735698699951172, 0.22838926315307617, 0.0)),
    Vector((-0.34315362572669983, 0.9392792582511902, 0.0)),
    Vector((-1.0, 0.0, 0.0)),
    Vector((0.10130790621042252, 0.9948551058769226, 0.0)),
    Vector((-0.7066295146942139, 0.7075836658477783, 0.0)),
    Vector((-0.9912516474723816, 0.1319856345653534, 0.0)),
    Vector((0.0897066593170166, 0.995968222618103, 0.0)),
    Vector((0.9958641529083252, -0.09085401147603989, 0.0)),
    Vector((0.07799188047647476, 0.9969539642333984, 0.0)),
    Vector((-0.9956868290901184, 0.09277855604887009, 0.0))
]
holesInfo = None
firstVertIndex = 62
numPolygonVerts = 62
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
