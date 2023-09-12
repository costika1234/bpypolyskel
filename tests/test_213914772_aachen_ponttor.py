import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((1.7455451488494873, 1.8145079612731934, 0.0)),
    Vector((7.277791500091553, 6.066917419433594, 0.0)),
    Vector((7.876060962677002, 6.979738235473633, 0.0)),
    Vector((8.094252586364746, 7.5808634757995605, 0.0)),
    Vector((8.059059143066406, 8.616134643554688, 0.0)),
    Vector((7.714171409606934, 9.940835952758789, 0.0)),
    Vector((6.707667827606201, 10.931578636169434, 0.0)),
    Vector((5.490011215209961, 11.354591369628906, 0.0)),
    Vector((4.286431789398193, 11.465909004211426, 0.0)),
    Vector((3.413660764694214, 11.187609672546387, 0.0)),
    Vector((-5.208478927612305, 4.3414626121521, 0.0)),
    Vector((-5.820828437805176, 3.4731714725494385, 0.0)),
    Vector((-6.123484134674072, 2.4267685413360596, 0.0)),
    Vector((-5.975677967071533, 1.0241427421569824, 0.0)),
    Vector((-5.067713737487793, -0.3005601465702057, 0.0)),
    Vector((-3.7655935287475586, -1.1354573965072632, 0.0)),
    Vector((-2.24527907371521, -1.2245138883590698, 0.0)),
    Vector((-1.0205813646316528, -0.8126322031021118, 0.0)),
    Vector((0.0, 0.0, 5.623648166656494)),
    Vector((1.7455451488494873, 1.8145079612731934, 5.623648166656494)),
    Vector((7.277791500091553, 6.066917419433594, 5.623648166656494)),
    Vector((7.876060962677002, 6.979738235473633, 5.623648166656494)),
    Vector((8.094252586364746, 7.5808634757995605, 5.623648166656494)),
    Vector((8.059059143066406, 8.616134643554688, 5.623648166656494)),
    Vector((7.714171409606934, 9.940835952758789, 5.623648166656494)),
    Vector((6.707667827606201, 10.931578636169434, 5.623648166656494)),
    Vector((5.490011215209961, 11.354591369628906, 5.623648166656494)),
    Vector((4.286431789398193, 11.465909004211426, 5.623648166656494)),
    Vector((3.413660764694214, 11.187609672546387, 5.623648166656494)),
    Vector((-5.208478927612305, 4.3414626121521, 5.623648166656494)),
    Vector((-5.820828437805176, 3.4731714725494385, 5.623648166656494)),
    Vector((-6.123484134674072, 2.4267685413360596, 5.623648166656494)),
    Vector((-5.975677967071533, 1.0241427421569824, 5.623648166656494)),
    Vector((-5.067713737487793, -0.3005601465702057, 5.623648166656494)),
    Vector((-3.7655935287475586, -1.1354573965072632, 5.623648166656494)),
    Vector((-2.24527907371521, -1.2245138883590698, 5.623648166656494)),
    Vector((-1.0205813646316528, -0.8126322031021118, 5.623648166656494)),
]
unitVectors = [
    Vector((0.6932790875434875, 0.7206690907478333, 0.0)),
    Vector((0.7928431630134583, 0.6094257831573486, 0.0)),
    Vector((0.5481637716293335, 0.8363710641860962, 0.0)),
    Vector((0.34119144082069397, 0.9399938583374023, 0.0)),
    Vector((-0.033974792808294296, 0.9994226098060608, 0.0)),
    Vector((-0.25195229053497314, 0.9677397012710571, 0.0)),
    Vector((-0.7126646041870117, 0.7015049457550049, 0.0)),
    Vector((-0.9446219205856323, 0.32816073298454285, 0.0)),
    Vector((-0.9957501888275146, 0.0920957550406456, 0.0)),
    Vector((-0.9527365565299988, -0.3037978410720825, 0.0)),
    Vector((-0.7831482887268066, -0.6218349933624268, 0.0)),
    Vector((-0.5763301849365234, -0.8172169327735901, 0.0)),
    Vector((-0.27784597873687744, -0.9606257081031799, 0.0)),
    Vector((0.10479792952537537, -0.9944935441017151, 0.0)),
    Vector((0.565357506275177, -0.8248460292816162, 0.0)),
    Vector((0.8418189883232117, -0.5397599935531616, 0.0)),
    Vector((0.9982886910438538, -0.058477431535720825, 0.0)),
    Vector((0.9478326439857483, 0.31876838207244873, 0.0)),
    Vector((0.7822998762130737, 0.6229019165039062, 0.0)),
]
holesInfo = None
firstVertIndex = 19
numPolygonVerts = 19
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
