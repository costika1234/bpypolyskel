import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((8.382881164550781, 20.460529327392578, 0.0)),
    Vector((8.454591751098633, 20.427133560180664, 0.0)),
    Vector((9.300764083862305, 22.430885314941406, 0.0)),
    Vector((9.207541465759277, 22.475412368774414, 0.0)),
    Vector((9.594772338867188, 23.421628952026367, 0.0)),
    Vector((8.626689910888672, 23.822378158569336, 0.0)),
    Vector((8.712740898132324, 24.02275276184082, 0.0)),
    Vector((6.934337139129639, 24.779722213745117, 0.0)),
    Vector((6.869798183441162, 24.635007858276367, 0.0)),
    Vector((3.4922661781311035, 26.059894561767578, 0.0)),
    Vector((3.556804656982422, 26.204608917236328, 0.0)),
    Vector((1.0182785987854004, 27.284406661987305, 0.0)),
    Vector((0.9322268962860107, 27.08403205871582, 0.0)),
    Vector((-3.485093116760254, 28.954200744628906, 0.0)),
    Vector((-3.3990414142608643, 29.15457534790039, 0.0)),
    Vector((-5.378229141235352, 29.989473342895508, 0.0)),
    Vector((-5.442768096923828, 29.83362579345703, 0.0)),
    Vector((-10.053699493408203, 31.78172492980957, 0.0)),
    Vector((-9.938963890075684, 32.04888916015625, 0.0)),
    Vector((-11.724534034729004, 32.794734954833984, 0.0)),
    Vector((-11.817756652832031, 32.57209777832031, 0.0)),
    Vector((-16.57210350036621, 34.57585906982422, 0.0)),
    Vector((-16.50039291381836, 34.753971099853516, 0.0)),
    Vector((-18.393524169921875, 35.555477142333984, 0.0)),
    Vector((-18.458065032958984, 35.38849639892578, 0.0)),
    Vector((-22.832345962524414, 37.236419677734375, 0.0)),
    Vector((-22.746292114257812, 37.43679428100586, 0.0)),
    Vector((-24.76849937438965, 38.29396057128906, 0.0)),
    Vector((-24.85455322265625, 38.09358596801758, 0.0)),
    Vector((-29.365076065063477, 39.99717330932617, 0.0)),
    Vector((-29.279024124145508, 40.20867919921875, 0.0)),
    Vector((-30.88531494140625, 40.88773727416992, 0.0)),
    Vector((-30.83511734008789, 40.99905776977539, 0.0)),
    Vector((-32.60633850097656, 41.74490737915039, 0.0)),
    Vector((-32.64219665527344, 41.655853271484375, 0.0)),
    Vector((-34.958412170410156, 42.11227798461914, 0.0)),
    Vector((-37.24594497680664, 42.2904052734375, 0.0)),
    Vector((-39.53348159790039, 42.10117721557617, 0.0)),
    Vector((-41.76365280151367, 41.52233123779297, 0.0)),
    Vector((-43.864749908447266, 40.58726501464844, 0.0)),
    Vector((-45.7794075012207, 39.32937240600586, 0.0)),
    Vector((-47.464595794677734, 37.75978088378906, 0.0)),
    Vector((-48.862953186035156, 35.93415451049805, 0.0)),
    Vector((-49.94578552246094, 33.89701843261719, 0.0)),
    Vector((-50.684417724609375, 31.71516227722168, 0.0)),
    Vector((-51.05015563964844, 29.44424819946289, 0.0)),
    Vector((-51.03583526611328, 27.139934539794922, 0.0)),
    Vector((-50.641456604003906, 24.8690128326416, 0.0)),
    Vector((-49.881351470947266, 22.69827651977539, 0.0)),
    Vector((-48.769866943359375, 20.683382034301758, 0.0)),
    Vector((-47.342857360839844, 18.86886215209961, 0.0)),
    Vector((-43.57094192504883, 15.68509292602539, 0.0)),
    Vector((-43.70002365112305, 15.384531021118164, 0.0)),
    Vector((-41.828399658203125, 14.594147682189941, 0.0)),
    Vector((-41.76386260986328, 14.738862991333008, 0.0)),
    Vector((-35.596824645996094, 12.133942604064941, 0.0)),
    Vector((-35.67570877075195, 11.944700241088867, 0.0)),
    Vector((-33.832767486572266, 11.165451049804688, 0.0)),
    Vector((-33.72520065307617, 11.421485900878906, 0.0)),
    Vector((-29.257678985595703, 9.529027938842773, 0.0)),
    Vector((-29.358074188232422, 9.284125328063965, 0.0)),
    Vector((-27.38605308532715, 8.44921875, 0.0)),
    Vector((-27.314342498779297, 8.62732982635498, 0.0)),
    Vector((-22.689054489135742, 6.679217338562012, 0.0)),
    Vector((-22.760765075683594, 6.501106262207031, 0.0)),
    Vector((-20.853281021118164, 5.688466548919678, 0.0)),
    Vector((-20.803083419799805, 5.810917854309082, 0.0)),
    Vector((-16.299699783325195, 3.907338857650757, 0.0)),
    Vector((-16.364238739013672, 3.7514917850494385, 0.0)),
    Vector((-14.542804718017578, 2.983381986618042, 0.0)),
    Vector((-14.45675277709961, 3.1837568283081055, 0.0)),
    Vector((-9.738232612609863, 1.1911274194717407, 0.0)),
    Vector((-9.824285507202148, 0.9796205163002014, 0.0)),
    Vector((-7.400484561920166, -0.033390749245882034, 0.0)),
    Vector((-7.350287437438965, 0.08906061947345734, 0.0)),
    Vector((-3.8938608169555664, -1.3803602457046509, 0.0)),
    Vector((-3.958400011062622, -1.5250755548477173, 0.0)),
    Vector((-2.029397487640381, -2.3377089500427246, 0.0)),
    Vector((-1.9648581743240356, -2.170729637145996, 0.0)),
    Vector((-1.0469683408737183, -2.5603482723236084, 0.0)),
    Vector((-0.688417375087738, -1.6809242963790894, 0.0)),
    Vector((-0.580852210521698, -1.7254520654678345, 0.0)),
    Vector((0.12190721184015274, -0.05565974488854408, 0.0)),
    Vector((0.0, 0.0, 10.0)),
    Vector((8.382881164550781, 20.460529327392578, 10.0)),
    Vector((8.454591751098633, 20.427133560180664, 10.0)),
    Vector((9.300764083862305, 22.430885314941406, 10.0)),
    Vector((9.207541465759277, 22.475412368774414, 10.0)),
    Vector((9.594772338867188, 23.421628952026367, 10.0)),
    Vector((8.626689910888672, 23.822378158569336, 10.0)),
    Vector((8.712740898132324, 24.02275276184082, 10.0)),
    Vector((6.934337139129639, 24.779722213745117, 10.0)),
    Vector((6.869798183441162, 24.635007858276367, 10.0)),
    Vector((3.4922661781311035, 26.059894561767578, 10.0)),
    Vector((3.556804656982422, 26.204608917236328, 10.0)),
    Vector((1.0182785987854004, 27.284406661987305, 10.0)),
    Vector((0.9322268962860107, 27.08403205871582, 10.0)),
    Vector((-3.485093116760254, 28.954200744628906, 10.0)),
    Vector((-3.3990414142608643, 29.15457534790039, 10.0)),
    Vector((-5.378229141235352, 29.989473342895508, 10.0)),
    Vector((-5.442768096923828, 29.83362579345703, 10.0)),
    Vector((-10.053699493408203, 31.78172492980957, 10.0)),
    Vector((-9.938963890075684, 32.04888916015625, 10.0)),
    Vector((-11.724534034729004, 32.794734954833984, 10.0)),
    Vector((-11.817756652832031, 32.57209777832031, 10.0)),
    Vector((-16.57210350036621, 34.57585906982422, 10.0)),
    Vector((-16.50039291381836, 34.753971099853516, 10.0)),
    Vector((-18.393524169921875, 35.555477142333984, 10.0)),
    Vector((-18.458065032958984, 35.38849639892578, 10.0)),
    Vector((-22.832345962524414, 37.236419677734375, 10.0)),
    Vector((-22.746292114257812, 37.43679428100586, 10.0)),
    Vector((-24.76849937438965, 38.29396057128906, 10.0)),
    Vector((-24.85455322265625, 38.09358596801758, 10.0)),
    Vector((-29.365076065063477, 39.99717330932617, 10.0)),
    Vector((-29.279024124145508, 40.20867919921875, 10.0)),
    Vector((-30.88531494140625, 40.88773727416992, 10.0)),
    Vector((-30.83511734008789, 40.99905776977539, 10.0)),
    Vector((-32.60633850097656, 41.74490737915039, 10.0)),
    Vector((-32.64219665527344, 41.655853271484375, 10.0)),
    Vector((-34.958412170410156, 42.11227798461914, 10.0)),
    Vector((-37.24594497680664, 42.2904052734375, 10.0)),
    Vector((-39.53348159790039, 42.10117721557617, 10.0)),
    Vector((-41.76365280151367, 41.52233123779297, 10.0)),
    Vector((-43.864749908447266, 40.58726501464844, 10.0)),
    Vector((-45.7794075012207, 39.32937240600586, 10.0)),
    Vector((-47.464595794677734, 37.75978088378906, 10.0)),
    Vector((-48.862953186035156, 35.93415451049805, 10.0)),
    Vector((-49.94578552246094, 33.89701843261719, 10.0)),
    Vector((-50.684417724609375, 31.71516227722168, 10.0)),
    Vector((-51.05015563964844, 29.44424819946289, 10.0)),
    Vector((-51.03583526611328, 27.139934539794922, 10.0)),
    Vector((-50.641456604003906, 24.8690128326416, 10.0)),
    Vector((-49.881351470947266, 22.69827651977539, 10.0)),
    Vector((-48.769866943359375, 20.683382034301758, 10.0)),
    Vector((-47.342857360839844, 18.86886215209961, 10.0)),
    Vector((-43.57094192504883, 15.68509292602539, 10.0)),
    Vector((-43.70002365112305, 15.384531021118164, 10.0)),
    Vector((-41.828399658203125, 14.594147682189941, 10.0)),
    Vector((-41.76386260986328, 14.738862991333008, 10.0)),
    Vector((-35.596824645996094, 12.133942604064941, 10.0)),
    Vector((-35.67570877075195, 11.944700241088867, 10.0)),
    Vector((-33.832767486572266, 11.165451049804688, 10.0)),
    Vector((-33.72520065307617, 11.421485900878906, 10.0)),
    Vector((-29.257678985595703, 9.529027938842773, 10.0)),
    Vector((-29.358074188232422, 9.284125328063965, 10.0)),
    Vector((-27.38605308532715, 8.44921875, 10.0)),
    Vector((-27.314342498779297, 8.62732982635498, 10.0)),
    Vector((-22.689054489135742, 6.679217338562012, 10.0)),
    Vector((-22.760765075683594, 6.501106262207031, 10.0)),
    Vector((-20.853281021118164, 5.688466548919678, 10.0)),
    Vector((-20.803083419799805, 5.810917854309082, 10.0)),
    Vector((-16.299699783325195, 3.907338857650757, 10.0)),
    Vector((-16.364238739013672, 3.7514917850494385, 10.0)),
    Vector((-14.542804718017578, 2.983381986618042, 10.0)),
    Vector((-14.45675277709961, 3.1837568283081055, 10.0)),
    Vector((-9.738232612609863, 1.1911274194717407, 10.0)),
    Vector((-9.824285507202148, 0.9796205163002014, 10.0)),
    Vector((-7.400484561920166, -0.033390749245882034, 10.0)),
    Vector((-7.350287437438965, 0.08906061947345734, 10.0)),
    Vector((-3.8938608169555664, -1.3803602457046509, 10.0)),
    Vector((-3.958400011062622, -1.5250755548477173, 10.0)),
    Vector((-2.029397487640381, -2.3377089500427246, 10.0)),
    Vector((-1.9648581743240356, -2.170729637145996, 10.0)),
    Vector((-1.0469683408737183, -2.5603482723236084, 10.0)),
    Vector((-0.688417375087738, -1.6809242963790894, 10.0)),
    Vector((-0.580852210521698, -1.7254520654678345, 10.0)),
    Vector((0.12190721184015274, -0.05565974488854408, 10.0)),
]
unitVectors = [
    Vector((0.3791234791278839, 0.9253461360931396, 0.0)),
    Vector((0.9065179228782654, -0.42216724157333374, 0.0)),
    Vector((0.3890281319618225, 0.9212258458137512, 0.0)),
    Vector((-0.9023513793945312, 0.43100109696388245, 0.0)),
    Vector((0.37875205278396606, 0.9254982471466064, 0.0)),
    Vector((-0.9239616990089417, 0.38248491287231445, 0.0)),
    Vector((0.39460164308547974, 0.9188522696495056, 0.0)),
    Vector((-0.9201170206069946, 0.39164361357688904, 0.0)),
    Vector((-0.4073052704334259, -0.9132921695709229, 0.0)),
    Vector((-0.9213651418685913, 0.3886983096599579, 0.0)),
    Vector((0.40730276703834534, 0.9132932424545288, 0.0)),
    Vector((-0.9202103018760681, 0.39142438769340515, 0.0)),
    Vector((-0.3946044147014618, -0.9188511371612549, 0.0)),
    Vector((-0.9208698272705078, 0.389870285987854, 0.0)),
    Vector((0.3946044147014618, 0.9188511371612549, 0.0)),
    Vector((-0.92137610912323, 0.3886721134185791, 0.0)),
    Vector((-0.3826064467430115, -0.9239113926887512, 0.0)),
    Vector((-0.9211592674255371, 0.38918590545654297, 0.0)),
    Vector((0.3946067690849304, 0.9188500642776489, 0.0)),
    Vector((-0.9227355718612671, 0.3854334354400635, 0.0)),
    Vector((-0.3862285614013672, -0.9224031567573547, 0.0)),
    Vector((-0.9215014576911926, 0.3883748948574066, 0.0)),
    Vector((0.37348097562789917, 0.9276378750801086, 0.0)),
    Vector((-0.920868456363678, 0.3898734748363495, 0.0)),
    Vector((-0.36052361130714417, -0.9327501058578491, 0.0)),
    Vector((-0.921173632144928, 0.38915154337882996, 0.0)),
    Vector((0.39461269974708557, 0.9188475012779236, 0.0)),
    Vector((-0.9207028150558472, 0.3902643620967865, 0.0)),
    Vector((-0.39461269974708557, -0.9188475012779236, 0.0)),
    Vector((-0.921312153339386, 0.3888236880302429, 0.0)),
    Vector((0.3768569231033325, 0.9262714982032776, 0.0)),
    Vector((-0.9210755228996277, 0.38938388228416443, 0.0)),
    Vector((0.41106855869293213, 0.9116044044494629, 0.0)),
    Vector((-0.9216219782829285, 0.38808897137641907, 0.0)),
    Vector((-0.37351348996162415, -0.9276247024536133, 0.0)),
    Vector((-0.9811322093009949, 0.19333823025226593, 0.0)),
    Vector((-0.9969819188117981, 0.07763372361660004, 0.0)),
    Vector((-0.9965960383415222, -0.08243974298238754, 0.0)),
    Vector((-0.9679279923439026, -0.25122788548469543, 0.0)),
    Vector((-0.9136105179786682, -0.4065905809402466, 0.0)),
    Vector((-0.8357677459716797, -0.5490830540657043, 0.0)),
    Vector((-0.7317590117454529, -0.6815634369850159, 0.0)),
    Vector((-0.6080783009529114, -0.7938770055770874, 0.0)),
    Vector((-0.4693593382835388, -0.883007287979126, 0.0)),
    Vector((-0.3206576406955719, -0.9471951723098755, 0.0)),
    Vector((-0.15900424122810364, -0.9872779250144958, 0.0)),
    Vector((0.006214473396539688, -0.999980628490448, 0.0)),
    Vector((0.1711035668849945, -0.985253095626831, 0.0)),
    Vector((0.3304849863052368, -0.9438112378120422, 0.0)),
    Vector((0.4830169081687927, -0.8756110668182373, 0.0)),
    Vector((0.6181737780570984, -0.7860414385795593, 0.0)),
    Vector((0.7641701102256775, -0.6450148224830627, 0.0)),
    Vector((-0.39461520314216614, -0.9188465476036072, 0.0)),
    Vector((0.9212244749069214, -0.38903138041496277, 0.0)),
    Vector((0.4072929620742798, 0.9132975339889526, 0.0)),
    Vector((0.9211928844451904, -0.389106422662735, 0.0)),
    Vector((-0.38475310802459717, -0.923019528388977, 0.0)),
    Vector((0.921049177646637, -0.3894463777542114, 0.0)),
    Vector((0.3873310387134552, 0.9219407439231873, 0.0)),
    Vector((0.9207931160926819, -0.3900512158870697, 0.0)),
    Vector((-0.3793052136898041, -0.9252716302871704, 0.0)),
    Vector((0.9208683371543884, -0.3898736238479614, 0.0)),
    Vector((0.37348270416259766, 0.9276371598243713, 0.0)),
    Vector((0.9215909838676453, -0.38816240429878235, 0.0)),
    Vector((-0.37348270416259766, -0.9276371598243713, 0.0)),
    Vector((0.9199904203414917, -0.39194077253341675, 0.0)),
    Vector((0.3793052136898041, 0.9252716302871704, 0.0)),
    Vector((0.9210919141769409, -0.38934528827667236, 0.0)),
    Vector((-0.38260746002197266, -0.9239110350608826, 0.0)),
    Vector((0.9214198589324951, -0.38856837153434753, 0.0)),
    Vector((0.39460495114326477, 0.9188508987426758, 0.0)),
    Vector((0.9212239980697632, -0.3890325725078583, 0.0)),
    Vector((-0.3768589496612549, -0.9262706637382507, 0.0)),
    Vector((0.9226582050323486, -0.38561877608299255, 0.0)),
    Vector((0.3793019652366638, 0.9252729415893555, 0.0)),
    Vector((0.9202888011932373, -0.3912397623062134, 0.0)),
    Vector((-0.4073042869567871, -0.9132925868034363, 0.0)),
    Vector((0.9215632677078247, -0.3882281482219696, 0.0)),
    Vector((0.3605187237262726, 0.9327519536018372, 0.0)),
    Vector((0.9205057621002197, -0.39072901010513306, 0.0)),
    Vector((0.3775380849838257, 0.9259939789772034, 0.0)),
    Vector((0.9239619970321655, -0.38248410820961, 0.0)),
    Vector((0.3879111707210541, 0.9216967821121216, 0.0)),
    Vector((-0.909669816493988, 0.41533219814300537, 0.0)),
]
holesInfo = None
firstVertIndex = 84
numPolygonVerts = 84
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
