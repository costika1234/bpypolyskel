import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((2.009035348892212, -1.8924310207366943, 0.0)),
    Vector((-2.1657705307006836, -6.356342315673828, 0.0)),
    Vector((-1.2966127395629883, -7.168974876403809, 0.0)),
    Vector((-1.4889678955078125, -8.861031532287598, 0.0)),
    Vector((0.22085176408290863, -10.486295700073242, 0.0)),
    Vector((1.8523049354553223, -10.107809066772461, 0.0)),
    Vector((2.5861029624938965, -10.786857604980469, 0.0)),
    Vector((2.7855818271636963, -10.575350761413574, 0.0)),
    Vector((8.406623840332031, -16.01886749267578, 0.0)),
    Vector((9.054930686950684, -15.306422233581543, 0.0)),
    Vector((9.154669761657715, -15.228498458862305, 0.0)),
    Vector((9.539380073547363, -15.451136589050293, 0.0)),
    Vector((9.567876815795898, -15.373212814331055, 0.0)),
    Vector((9.995332717895508, -15.440004348754883, 0.0)),
    Vector((10.052327156066895, -15.506795883178711, 0.0)),
    Vector((10.437036514282227, -15.440003395080566, 0.0)),
    Vector((10.558149337768555, -15.495662689208984, 0.0)),
    Vector((10.86449146270752, -15.373210906982422, 0.0)),
    Vector((11.021224975585938, -15.384342193603516, 0.0)),
    Vector((11.277698516845703, -15.217362403869629, 0.0)),
    Vector((11.455804824829102, -15.161702156066895, 0.0)),
    Vector((11.854762077331543, -14.872270584106445, 0.0)),
    Vector((12.004371643066406, -14.738686561584473, 0.0)),
    Vector((12.296464920043945, -14.404727935791016, 0.0)),
    Vector((12.531564712524414, -14.037372589111328, 0.0)),
    Vector((12.702546119689941, -13.636622428894043, 0.0)),
    Vector((12.723917961120605, -13.391719818115234, 0.0)),
    Vector((12.802284240722656, -13.202476501464844, 0.0)),
    Vector((12.837904930114746, -12.768329620361328, 0.0)),
    Vector((12.752412796020508, -12.478899955749512, 0.0)),
    Vector((12.788033485412598, -12.334184646606445, 0.0)),
    Vector((12.674044609069824, -11.91117000579834, 0.0)),
    Vector((12.488813400268555, -11.510420799255371, 0.0)),
    Vector((17.782114028930664, -5.866507530212402, 0.0)),
    Vector((17.682373046875, -5.755188465118408, 0.0)),
    Vector((19.491924285888672, -3.818222761154175, 0.0)),
    Vector((15.374101638793945, -0.022241652011871338, 0.0)),
    Vector((16.68495750427246, 1.4137837886810303, 0.0)),
    Vector((12.517271041870117, 5.309954643249512, 0.0)),
    Vector((11.213540077209473, 3.9073259830474854, 0.0)),
    Vector((7.05298376083374, 7.803501129150391, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((2.009035348892212, -1.8924310207366943, 4.73146915435791)),
    Vector((-2.1657705307006836, -6.356342315673828, 4.73146915435791)),
    Vector((-1.2966127395629883, -7.168974876403809, 4.73146915435791)),
    Vector((-1.4889678955078125, -8.861031532287598, 4.73146915435791)),
    Vector((0.22085176408290863, -10.486295700073242, 4.73146915435791)),
    Vector((1.8523049354553223, -10.107809066772461, 4.73146915435791)),
    Vector((2.5861029624938965, -10.786857604980469, 4.73146915435791)),
    Vector((2.7855818271636963, -10.575350761413574, 4.73146915435791)),
    Vector((8.406623840332031, -16.01886749267578, 4.73146915435791)),
    Vector((9.054930686950684, -15.306422233581543, 4.73146915435791)),
    Vector((9.154669761657715, -15.228498458862305, 4.73146915435791)),
    Vector((9.539380073547363, -15.451136589050293, 4.73146915435791)),
    Vector((9.567876815795898, -15.373212814331055, 4.73146915435791)),
    Vector((9.995332717895508, -15.440004348754883, 4.73146915435791)),
    Vector((10.052327156066895, -15.506795883178711, 4.73146915435791)),
    Vector((10.437036514282227, -15.440003395080566, 4.73146915435791)),
    Vector((10.558149337768555, -15.495662689208984, 4.73146915435791)),
    Vector((10.86449146270752, -15.373210906982422, 4.73146915435791)),
    Vector((11.021224975585938, -15.384342193603516, 4.73146915435791)),
    Vector((11.277698516845703, -15.217362403869629, 4.73146915435791)),
    Vector((11.455804824829102, -15.161702156066895, 4.73146915435791)),
    Vector((11.854762077331543, -14.872270584106445, 4.73146915435791)),
    Vector((12.004371643066406, -14.738686561584473, 4.73146915435791)),
    Vector((12.296464920043945, -14.404727935791016, 4.73146915435791)),
    Vector((12.531564712524414, -14.037372589111328, 4.73146915435791)),
    Vector((12.702546119689941, -13.636622428894043, 4.73146915435791)),
    Vector((12.723917961120605, -13.391719818115234, 4.73146915435791)),
    Vector((12.802284240722656, -13.202476501464844, 4.73146915435791)),
    Vector((12.837904930114746, -12.768329620361328, 4.73146915435791)),
    Vector((12.752412796020508, -12.478899955749512, 4.73146915435791)),
    Vector((12.788033485412598, -12.334184646606445, 4.73146915435791)),
    Vector((12.674044609069824, -11.91117000579834, 4.73146915435791)),
    Vector((12.488813400268555, -11.510420799255371, 4.73146915435791)),
    Vector((17.782114028930664, -5.866507530212402, 4.73146915435791)),
    Vector((17.682373046875, -5.755188465118408, 4.73146915435791)),
    Vector((19.491924285888672, -3.818222761154175, 4.73146915435791)),
    Vector((15.374101638793945, -0.022241652011871338, 4.73146915435791)),
    Vector((16.68495750427246, 1.4137837886810303, 4.73146915435791)),
    Vector((12.517271041870117, 5.309954643249512, 4.73146915435791)),
    Vector((11.213540077209473, 3.9073259830474854, 4.73146915435791)),
    Vector((7.05298376083374, 7.803501129150391, 4.73146915435791)),
    Vector((0.0, 0.0, 4.73146915435791))
]
unitVectors = [
    Vector((-0.6830601692199707, -0.7303621172904968, 0.0)),
    Vector((0.7304602265357971, -0.6829551458358765, 0.0)),
    Vector((-0.11295374482870102, -0.9936002492904663, 0.0)),
    Vector((0.7248013615608215, -0.6889578104019165, 0.0)),
    Vector((0.9741292595863342, 0.22599171102046967, 0.0)),
    Vector((0.7339571714401245, -0.679195761680603, 0.0)),
    Vector((0.6861187219619751, 0.7274895906448364, 0.0)),
    Vector((0.7183594703674316, -0.6956720352172852, 0.0)),
    Vector((0.673030436038971, 0.7396148443222046, 0.0)),
    Vector((0.78801429271698, 0.615656852722168, 0.0)),
    Vector((0.8655129671096802, -0.5008864998817444, 0.0)),
    Vector((0.34345442056655884, 0.9391692876815796, 0.0)),
    Vector((0.9880114793777466, -0.15438036620616913, 0.0)),
    Vector((0.6491126418113708, -0.760692298412323, 0.0)),
    Vector((0.9852607846260071, 0.17105905711650848, 0.0)),
    Vector((0.9086402058601379, -0.4175798296928406, 0.0)),
    Vector((0.9285656213760376, 0.3711683750152588, 0.0)),
    Vector((0.9974875450134277, -0.07084202766418457, 0.0)),
    Vector((0.8380372524261475, 0.5456129312515259, 0.0)),
    Vector((0.9544768333435059, 0.2982848584651947, 0.0)),
    Vector((0.8094295859336853, 0.5872169733047485, 0.0)),
    Vector((0.745927631855011, 0.6660270094871521, 0.0)),
    Vector((0.6583507061004639, 0.752711296081543, 0.0)),
    Vector((0.53904128074646, 0.842279314994812, 0.0)),
    Vector((0.39242830872535706, 0.9197825193405151, 0.0)),
    Vector((0.08693629503250122, 0.9962139129638672, 0.0)),
    Vector((0.3825964629650116, 0.9239155650138855, 0.0)),
    Vector((0.08177277445793152, 0.9966509342193604, 0.0)),
    Vector((-0.2832816243171692, 0.9590368270874023, 0.0)),
    Vector((0.2390093058347702, 0.9710173010826111, 0.0)),
    Vector((-0.26018697023391724, 0.9655582904815674, 0.0)),
    Vector((-0.41956233978271484, 0.907726526260376, 0.0)),
    Vector((0.6840877532958984, 0.7293996810913086, 0.0)),
    Vector((-0.667313814163208, 0.7447766065597534, 0.0)),
    Vector((0.6826642155647278, 0.7307321429252625, 0.0)),
    Vector((-0.7352558374404907, 0.677789568901062, 0.0)),
    Vector((0.6741858720779419, 0.7385618090629578, 0.0)),
    Vector((-0.7305015921592712, 0.6829110383987427, 0.0)),
    Vector((-0.6808127760887146, -0.7324575185775757, 0.0)),
    Vector((-0.7299175262451172, 0.6835351586341858, 0.0)),
    Vector((-0.6705302000045776, -0.7418822050094604, 0.0)),
    Vector((0.7279152274131775, -0.685667097568512, 0.0))
]
holesInfo = None
firstVertIndex = 42
numPolygonVerts = 42
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
