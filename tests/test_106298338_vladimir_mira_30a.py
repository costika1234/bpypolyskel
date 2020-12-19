import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((16.771930694580078, 0.5788941979408264, 0.0)),
    Vector((17.175132751464844, -7.2913923263549805, 0.0)),
    Vector((31.100072860717773, -6.601132869720459, 0.0)),
    Vector((30.541778564453125, 1.4583942890167236, 0.0)),
    Vector((61.67277908325195, 3.306633234024048, 0.0)),
    Vector((75.24404907226562, 8.305095672607422, 0.0)),
    Vector((59.71820068359375, 57.1629753112793, 0.0)),
    Vector((47.325538635253906, 53.27777099609375, 0.0)),
    Vector((59.452117919921875, 11.755751609802246, 0.0)),
    Vector((43.84634017944336, 10.219353675842285, 0.0)),
    Vector((43.59819793701172, 13.993082046508789, 0.0)),
    Vector((7.498968601226807, 11.733080863952637, 0.0)),
    Vector((6.86008882522583, 17.68867301940918, 0.0)),
    Vector((-0.5396273136138916, 17.466028213500977, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((16.771930694580078, 0.5788941979408264, 6.169360637664795)),
    Vector((17.175132751464844, -7.2913923263549805, 6.169360637664795)),
    Vector((31.100072860717773, -6.601132869720459, 6.169360637664795)),
    Vector((30.541778564453125, 1.4583942890167236, 6.169360637664795)),
    Vector((61.67277908325195, 3.306633234024048, 6.169360637664795)),
    Vector((75.24404907226562, 8.305095672607422, 6.169360637664795)),
    Vector((59.71820068359375, 57.1629753112793, 6.169360637664795)),
    Vector((47.325538635253906, 53.27777099609375, 6.169360637664795)),
    Vector((59.452117919921875, 11.755751609802246, 6.169360637664795)),
    Vector((43.84634017944336, 10.219353675842285, 6.169360637664795)),
    Vector((43.59819793701172, 13.993082046508789, 6.169360637664795)),
    Vector((7.498968601226807, 11.733080863952637, 6.169360637664795)),
    Vector((6.86008882522583, 17.68867301940918, 6.169360637664795)),
    Vector((-0.5396273136138916, 17.466028213500977, 6.169360637664795)),
    Vector((0.0, 0.0, 6.169360637664795))
]
unitVectors = [
    Vector((0.05116382613778114, -0.9986902475357056, 0.0)),
    Vector((0.9987736344337463, 0.04950922355055809, 0.0)),
    Vector((-0.06910574436187744, 0.9976093769073486, 0.0)),
    Vector((0.998242199420929, 0.059265367686748505, 0.0)),
    Vector((0.9383763074874878, 0.34561532735824585, 0.0)),
    Vector((-0.30285221338272095, 0.9530375599861145, 0.0)),
    Vector((-0.9542056322097778, -0.29915153980255127, 0.0)),
    Vector((0.28034067153930664, -0.9599005579948425, 0.0)),
    Vector((-0.9951886534690857, -0.09797690063714981, 0.0)),
    Vector((-0.06561350077390671, 0.9978451132774353, 0.0)),
    Vector((-0.9980460405349731, -0.06248292326927185, 0.0)),
    Vector((-0.10666196793317795, 0.9942953586578369, 0.0)),
    Vector((-0.9995477199554443, -0.030074680224061012, 0.0)),
    Vector((0.030881090089678764, -0.9995231628417969, 0.0)),
    Vector((0.9994048476219177, 0.03449511528015137, 0.0))
]
holesInfo = None
firstVertIndex = 15
numPolygonVerts = 15
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