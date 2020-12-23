import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((1.207115650177002, -0.5120695233345032, 0.0)),
    Vector((2.1281516551971436, 0.02226433902978897, 0.0)),
    Vector((3.558548927307129, -0.545464277267456, 0.0)),
    Vector((3.279447555541992, -1.2913050651550293, 0.0)),
    Vector((12.936384201049805, -5.187471866607666, 0.0)),
    Vector((13.055001258850098, -4.820117473602295, 0.0)),
    Vector((16.201881408691406, -6.133678436279297, 0.0)),
    Vector((18.183490753173828, -1.302405834197998, 0.0)),
    Vector((18.748672485351562, -1.4916468858718872, 0.0)),
    Vector((19.983686447143555, 1.4917200803756714, 0.0)),
    Vector((19.390594482421875, 1.714356780052185, 0.0)),
    Vector((21.44894790649414, 6.979776859283447, 0.0)),
    Vector((19.33475112915039, 7.803532600402832, 0.0)),
    Vector((18.818408966064453, 9.061441421508789, 0.0)),
    Vector((17.499656677246094, 8.53823471069336, 0.0)),
    Vector((14.54815673828125, 9.695948600769043, 0.0)),
    Vector((14.708638191223145, 10.274809837341309, 0.0)),
    Vector((12.475826263427734, 11.165360450744629, 0.0)),
    Vector((12.266501426696777, 10.575366020202637, 0.0)),
    Vector((9.601081848144531, 11.644027709960938, 0.0)),
    Vector((9.126607894897461, 12.913068771362305, 0.0)),
    Vector((7.793900489807129, 12.36760139465332, 0.0)),
    Vector((5.065685272216797, 13.458528518676758, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((1.207115650177002, -0.5120695233345032, 8.846820831298828)),
    Vector((2.1281516551971436, 0.02226433902978897, 8.846820831298828)),
    Vector((3.558548927307129, -0.545464277267456, 8.846820831298828)),
    Vector((3.279447555541992, -1.2913050651550293, 8.846820831298828)),
    Vector((12.936384201049805, -5.187471866607666, 8.846820831298828)),
    Vector((13.055001258850098, -4.820117473602295, 8.846820831298828)),
    Vector((16.201881408691406, -6.133678436279297, 8.846820831298828)),
    Vector((18.183490753173828, -1.302405834197998, 8.846820831298828)),
    Vector((18.748672485351562, -1.4916468858718872, 8.846820831298828)),
    Vector((19.983686447143555, 1.4917200803756714, 8.846820831298828)),
    Vector((19.390594482421875, 1.714356780052185, 8.846820831298828)),
    Vector((21.44894790649414, 6.979776859283447, 8.846820831298828)),
    Vector((19.33475112915039, 7.803532600402832, 8.846820831298828)),
    Vector((18.818408966064453, 9.061441421508789, 8.846820831298828)),
    Vector((17.499656677246094, 8.53823471069336, 8.846820831298828)),
    Vector((14.54815673828125, 9.695948600769043, 8.846820831298828)),
    Vector((14.708638191223145, 10.274809837341309, 8.846820831298828)),
    Vector((12.475826263427734, 11.165360450744629, 8.846820831298828)),
    Vector((12.266501426696777, 10.575366020202637, 8.846820831298828)),
    Vector((9.601081848144531, 11.644027709960938, 8.846820831298828)),
    Vector((9.126607894897461, 12.913068771362305, 8.846820831298828)),
    Vector((7.793900489807129, 12.36760139465332, 8.846820831298828)),
    Vector((5.065685272216797, 13.458528518676758, 8.846820831298828)),
    Vector((0.0, 0.0, 8.846820831298828))
]
unitVectors = [
    Vector((0.8649770021438599, 0.5018115639686584, 0.0)),
    Vector((0.9294660687446594, -0.3689076602458954, 0.0)),
    Vector((-0.350475013256073, -0.9365720748901367, 0.0)),
    Vector((0.9273667931556702, -0.37415340542793274, 0.0)),
    Vector((0.30727407336235046, 0.9516209959983826, 0.0)),
    Vector((0.9228309392929077, -0.3852052390575409, 0.0)),
    Vector((0.3794823884963989, 0.9251989722251892, 0.0)),
    Vector((0.9482560157775879, -0.31750667095184326, 0.0)),
    Vector((0.38248857855796814, 0.923960268497467, 0.0)),
    Vector((-0.9362112283706665, 0.35143786668777466, 0.0)),
    Vector((0.3640882074832916, 0.9313645362854004, 0.0)),
    Vector((-0.9317710399627686, 0.3630464971065521, 0.0)),
    Vector((-0.3797307312488556, 0.9250970482826233, 0.0)),
    Vector((-0.9295167326927185, -0.3687799572944641, 0.0)),
    Vector((-0.9309450387954712, 0.36515942215919495, 0.0)),
    Vector((0.2671595513820648, 0.9636522531509399, 0.0)),
    Vector((-0.9288454651832581, 0.37046733498573303, 0.0)),
    Vector((-0.33437010645866394, -0.9424418807029724, 0.0)),
    Vector((-0.9281769394874573, 0.37213921546936035, 0.0)),
    Vector((-0.3502067029476166, 0.9366724491119385, 0.0)),
    Vector((-0.9254814386367798, -0.37879276275634766, 0.0)),
    Vector((-0.9285188913345337, 0.3712853789329529, 0.0)),
    Vector((-0.35226550698280334, -0.935900092124939, 0.0)),
    Vector((0.9205927848815918, -0.39052391052246094, 0.0))
]
holesInfo = None
firstVertIndex = 24
numPolygonVerts = 24
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
    assert not bpypolyskel.checkEdgeCrossing(bpypolyskel.debugOutputs["skeleton"])