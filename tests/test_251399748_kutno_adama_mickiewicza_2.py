import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((0.6749575734138489, 0.30056267976760864, 0.0)),
    Vector((1.2476487159729004, 0.7681046724319458, 0.0)),
    Vector((1.6771669387817383, 1.358098030090332, 0.0)),
    Vector((1.943058967590332, 2.048279047012329, 0.0)),
    Vector((2.0112361907958984, 2.782987594604492, 0.0)),
    Vector((1.8953341245651245, 3.5176963806152344, 0.0)),
    Vector((1.5953530073165894, 4.18561315536499, 0.0)),
    Vector((1.1249282360076904, 4.753342151641846, 0.0)),
    Vector((0.5249664783477783, 5.187488079071045, 0.0)),
    Vector((-0.16362589597702026, 5.454655170440674, 0.0)),
    Vector((-0.8931246995925903, 5.521446704864502, 0.0)),
    Vector((-1.6226235628128052, 5.410127639770508, 0.0)),
    Vector((-2.9316282272338867, 9.762720108032227, 0.0)),
    Vector((-4.465620040893555, 9.29517936706543, 0.0)),
    Vector((-5.338290691375732, 9.606874465942383, 0.0)),
    Vector((-6.299592018127441, 9.46216106414795, 0.0)),
    Vector((-7.25407600402832, 9.139335632324219, 0.0)),
    Vector((-7.799496650695801, 8.30444049835205, 0.0)),
    Vector((-8.97896671295166, 7.9482197761535645, 0.0)),
    Vector((-8.822159767150879, 7.436150074005127, 0.0)),
    Vector((-14.83541488647461, 5.63278865814209, 0.0)),
    Vector((-15.29219913482666, 7.168998718261719, 0.0)),
    Vector((-16.376218795776367, 7.2803215980529785, 0.0)),
    Vector((-17.501148223876953, 7.057686805725098, 0.0)),
    Vector((-18.462451934814453, 6.623544216156006, 0.0)),
    Vector((-19.076051712036133, 6.022421360015869, 0.0)),
    Vector((-18.612449645996094, 4.497342586517334, 0.0)),
    Vector((-24.871150970458984, 2.616070508956909, 0.0)),
    Vector((-25.02113914489746, 3.117009162902832, 0.0)),
    Vector((-26.53468132019043, 2.671739101409912, 0.0)),
    Vector((-27.386899948120117, 2.8498549461364746, 0.0)),
    Vector((-28.266389846801758, 2.7162764072418213, 0.0)),
    Vector((-29.064069747924805, 2.315530776977539, 0.0)),
    Vector((-29.72539520263672, 1.703277587890625, 0.0)),
    Vector((-31.238937377929688, 1.2468770742416382, 0.0)),
    Vector((-29.875415802001953, -3.28383469581604, 0.0)),
    Vector((-30.570829391479492, -3.595525026321411, 0.0)),
    Vector((-31.15715789794922, -4.07419490814209, 0.0)),
    Vector((-31.600317001342773, -4.6864495277404785, 0.0)),
    Vector((-31.866214752197266, -5.398892402648926, 0.0)),
    Vector((-31.94121551513672, -6.155864715576172, 0.0)),
    Vector((-31.818500518798828, -6.901706218719482, 0.0)),
    Vector((-31.51170539855957, -7.591888904571533, 0.0)),
    Vector((-31.027647018432617, -8.181884765625, 0.0)),
    Vector((-30.414051055908203, -8.616035461425781, 0.0)),
    Vector((-29.705005645751953, -8.883206367492676, 0.0)),
    Vector((-28.95505142211914, -8.96113395690918, 0.0)),
    Vector((-28.205097198486328, -8.8386869430542, 0.0)),
    Vector((-27.95966148376465, -9.662452697753906, 0.0)),
    Vector((-24.85075569152832, -8.771913528442383, 0.0)),
    Vector((-24.38033676147461, -10.41944408416748, 0.0)),
    Vector((-15.394511222839355, -7.71441650390625, 0.0)),
    Vector((-14.228673934936523, -7.83687162399292, 0.0)),
    Vector((-13.049200057983398, -7.669895648956299, 0.0)),
    Vector((-11.924267768859863, -7.213488578796387, 0.0)),
    Vector((-11.215219497680664, -6.456517696380615, 0.0)),
    Vector((-2.2294070720672607, -3.7625982761383057, 0.0)),
    Vector((-2.6793782711029053, -2.148465394973755, 0.0)),
    Vector((0.409065306186676, -1.358097791671753, 0.0)),
    Vector((0.0, 0.0, 6.123260498046875)),
    Vector((0.6749575734138489, 0.30056267976760864, 6.123260498046875)),
    Vector((1.2476487159729004, 0.7681046724319458, 6.123260498046875)),
    Vector((1.6771669387817383, 1.358098030090332, 6.123260498046875)),
    Vector((1.943058967590332, 2.048279047012329, 6.123260498046875)),
    Vector((2.0112361907958984, 2.782987594604492, 6.123260498046875)),
    Vector((1.8953341245651245, 3.5176963806152344, 6.123260498046875)),
    Vector((1.5953530073165894, 4.18561315536499, 6.123260498046875)),
    Vector((1.1249282360076904, 4.753342151641846, 6.123260498046875)),
    Vector((0.5249664783477783, 5.187488079071045, 6.123260498046875)),
    Vector((-0.16362589597702026, 5.454655170440674, 6.123260498046875)),
    Vector((-0.8931246995925903, 5.521446704864502, 6.123260498046875)),
    Vector((-1.6226235628128052, 5.410127639770508, 6.123260498046875)),
    Vector((-2.9316282272338867, 9.762720108032227, 6.123260498046875)),
    Vector((-4.465620040893555, 9.29517936706543, 6.123260498046875)),
    Vector((-5.338290691375732, 9.606874465942383, 6.123260498046875)),
    Vector((-6.299592018127441, 9.46216106414795, 6.123260498046875)),
    Vector((-7.25407600402832, 9.139335632324219, 6.123260498046875)),
    Vector((-7.799496650695801, 8.30444049835205, 6.123260498046875)),
    Vector((-8.97896671295166, 7.9482197761535645, 6.123260498046875)),
    Vector((-8.822159767150879, 7.436150074005127, 6.123260498046875)),
    Vector((-14.83541488647461, 5.63278865814209, 6.123260498046875)),
    Vector((-15.29219913482666, 7.168998718261719, 6.123260498046875)),
    Vector((-16.376218795776367, 7.2803215980529785, 6.123260498046875)),
    Vector((-17.501148223876953, 7.057686805725098, 6.123260498046875)),
    Vector((-18.462451934814453, 6.623544216156006, 6.123260498046875)),
    Vector((-19.076051712036133, 6.022421360015869, 6.123260498046875)),
    Vector((-18.612449645996094, 4.497342586517334, 6.123260498046875)),
    Vector((-24.871150970458984, 2.616070508956909, 6.123260498046875)),
    Vector((-25.02113914489746, 3.117009162902832, 6.123260498046875)),
    Vector((-26.53468132019043, 2.671739101409912, 6.123260498046875)),
    Vector((-27.386899948120117, 2.8498549461364746, 6.123260498046875)),
    Vector((-28.266389846801758, 2.7162764072418213, 6.123260498046875)),
    Vector((-29.064069747924805, 2.315530776977539, 6.123260498046875)),
    Vector((-29.72539520263672, 1.703277587890625, 6.123260498046875)),
    Vector((-31.238937377929688, 1.2468770742416382, 6.123260498046875)),
    Vector((-29.875415802001953, -3.28383469581604, 6.123260498046875)),
    Vector((-30.570829391479492, -3.595525026321411, 6.123260498046875)),
    Vector((-31.15715789794922, -4.07419490814209, 6.123260498046875)),
    Vector((-31.600317001342773, -4.6864495277404785, 6.123260498046875)),
    Vector((-31.866214752197266, -5.398892402648926, 6.123260498046875)),
    Vector((-31.94121551513672, -6.155864715576172, 6.123260498046875)),
    Vector((-31.818500518798828, -6.901706218719482, 6.123260498046875)),
    Vector((-31.51170539855957, -7.591888904571533, 6.123260498046875)),
    Vector((-31.027647018432617, -8.181884765625, 6.123260498046875)),
    Vector((-30.414051055908203, -8.616035461425781, 6.123260498046875)),
    Vector((-29.705005645751953, -8.883206367492676, 6.123260498046875)),
    Vector((-28.95505142211914, -8.96113395690918, 6.123260498046875)),
    Vector((-28.205097198486328, -8.8386869430542, 6.123260498046875)),
    Vector((-27.95966148376465, -9.662452697753906, 6.123260498046875)),
    Vector((-24.85075569152832, -8.771913528442383, 6.123260498046875)),
    Vector((-24.38033676147461, -10.41944408416748, 6.123260498046875)),
    Vector((-15.394511222839355, -7.71441650390625, 6.123260498046875)),
    Vector((-14.228673934936523, -7.83687162399292, 6.123260498046875)),
    Vector((-13.049200057983398, -7.669895648956299, 6.123260498046875)),
    Vector((-11.924267768859863, -7.213488578796387, 6.123260498046875)),
    Vector((-11.215219497680664, -6.456517696380615, 6.123260498046875)),
    Vector((-2.2294070720672607, -3.7625982761383057, 6.123260498046875)),
    Vector((-2.6793782711029053, -2.148465394973755, 6.123260498046875)),
    Vector((0.409065306186676, -1.358097791671753, 6.123260498046875))
]
unitVectors = [
    Vector((0.9135192036628723, 0.4067956209182739, 0.0)),
    Vector((0.7746353149414062, 0.6324081420898438, 0.0)),
    Vector((0.5885589122772217, 0.8084542751312256, 0.0)),
    Vector((0.359494686126709, 0.9331471920013428, 0.0)),
    Vector((0.09239796549081802, 0.9957221746444702, 0.0)),
    Vector((-0.1558253914117813, 0.9877846837043762, 0.0)),
    Vector((-0.40970417857170105, 0.9122183918952942, 0.0)),
    Vector((-0.6380345225334167, 0.7700077295303345, 0.0)),
    Vector((-0.8101403713226318, 0.5862359404563904, 0.0)),
    Vector((-0.9322873950004578, 0.36171838641166687, 0.0)),
    Vector((-0.9958347678184509, 0.09117674827575684, 0.0)),
    Vector((-0.9885566234588623, -0.15085040032863617, 0.0)),
    Vector((-0.28799915313720703, 0.9576306343078613, 0.0)),
    Vector((-0.9565567970275879, -0.2915460765361786, 0.0)),
    Vector((-0.94173264503479, 0.33636221289634705, 0.0)),
    Vector((-0.9888579845428467, -0.14886176586151123, 0.0)),
    Vector((-0.9472854733467102, -0.3203907608985901, 0.0)),
    Vector((-0.5469175577163696, -0.8371865749359131, 0.0)),
    Vector((-0.9572930932044983, -0.2891193628311157, 0.0)),
    Vector((0.2928012013435364, -0.9561733603477478, 0.0)),
    Vector((-0.9578533172607422, -0.28725799918174744, 0.0)),
    Vector((-0.28501221537590027, 0.9585239291191101, 0.0)),
    Vector((-0.9947682023048401, 0.10215724259614944, 0.0)),
    Vector((-0.980972945690155, -0.1941443681716919, 0.0)),
    Vector((-0.911368727684021, -0.4115910232067108, 0.0)),
    Vector((-0.7143320441246033, -0.6998068690299988, 0.0)),
    Vector((0.2908444404602051, -0.9567703008651733, 0.0)),
    Vector((-0.9576719403266907, -0.2878618836402893, 0.0)),
    Vector((-0.2868330478668213, 0.9579805731773376, 0.0)),
    Vector((-0.9593464732170105, -0.28223082423210144, 0.0)),
    Vector((-0.978849470615387, 0.20458200573921204, 0.0)),
    Vector((-0.9886617064476013, -0.15015976130962372, 0.0)),
    Vector((-0.8935716152191162, -0.4489205777645111, 0.0)),
    Vector((-0.7338078022003174, -0.67935711145401, 0.0)),
    Vector((-0.9574183225631714, -0.28870436549186707, 0.0)),
    Vector((0.2881831228733063, -0.9575753211975098, 0.0)),
    Vector((-0.9125322103500366, -0.40900474786758423, 0.0)),
    Vector((-0.7746390104293823, -0.6324037313461304, 0.0)),
    Vector((-0.586338222026825, -0.8100663423538208, 0.0)),
    Vector((-0.3496607542037964, -0.9368763566017151, 0.0)),
    Vector((-0.09859715402126312, -0.9951274394989014, 0.0)),
    Vector((0.16234947741031647, -0.9867333173751831, 0.0)),
    Vector((0.4061906933784485, -0.9137883186340332, 0.0)),
    Vector((0.6342846751213074, -0.7730995416641235, 0.0)),
    Vector((0.8163254261016846, -0.5775921940803528, 0.0)),
    Vector((0.9357731342315674, -0.3526027500629425, 0.0)),
    Vector((0.9946447014808655, -0.10335332900285721, 0.0)),
    Vector((0.9869316816329956, 0.16113895177841187, 0.0)),
    Vector((0.28553929924964905, -0.9583670496940613, 0.0)),
    Vector((0.9613374471664429, 0.2753729820251465, 0.0)),
    Vector((0.2745570242404938, -0.9615707993507385, 0.0)),
    Vector((0.9575536251068115, 0.288254976272583, 0.0)),
    Vector((0.9945289492607117, -0.10446154326200485, 0.0)),
    Vector((0.990127444267273, 0.14017054438591003, 0.0)),
    Vector((0.9266378283500671, 0.37595513463020325, 0.0)),
    Vector((0.6836270689964294, 0.7298315167427063, 0.0)),
    Vector((0.9578797817230225, 0.2871694564819336, 0.0)),
    Vector((-0.2685307264328003, 0.9632711410522461, 0.0)),
    Vector((0.9687801003456116, 0.24792177975177765, 0.0)),
    Vector((-0.2884058952331543, 0.9575082063674927, 0.0))
]
holesInfo = None
firstVertIndex = 60
numPolygonVerts = 60
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
