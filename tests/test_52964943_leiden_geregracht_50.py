import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-0.17757786810398102, 0.44527795910835266, 0.0)),
    Vector((-3.5857081413269043, -1.0130060911178589, 0.0)),
    Vector((-3.4764297008514404, -1.3358327150344849, 0.0)),
    Vector((-8.796941757202148, -3.584479808807373, 0.0)),
    Vector((-8.954029083251953, -3.295048952102661, 0.0)),
    Vector((-11.986518859863281, -4.586348533630371, 0.0)),
    Vector((-11.85675048828125, -4.898043632507324, 0.0)),
    Vector((-12.06847858428955, -4.987098693847656, 0.0)),
    Vector((-11.911391258239746, -5.3767170906066895, 0.0)),
    Vector((-11.713323593139648, -5.298793792724609, 0.0)),
    Vector((-10.142447471618652, -9.005736351013184, 0.0)),
    Vector((-9.684842109680176, -8.805362701416016, 0.0)),
    Vector((-9.54824447631836, -9.150452613830566, 0.0)),
    Vector((-6.8231000900268555, -7.981602668762207, 0.0)),
    Vector((-6.966527938842773, -7.636512279510498, 0.0)),
    Vector((1.564052700996399, -4.007501602172852, 0.0)),
    Vector((-0.10927870124578476, -0.044527795165777206, 0.0)),
    Vector((0.0, 0.0, 2.9958152770996094)),
    Vector((-0.17757786810398102, 0.44527795910835266, 2.9958152770996094)),
    Vector((-3.5857081413269043, -1.0130060911178589, 2.9958152770996094)),
    Vector((-3.4764297008514404, -1.3358327150344849, 2.9958152770996094)),
    Vector((-8.796941757202148, -3.584479808807373, 2.9958152770996094)),
    Vector((-8.954029083251953, -3.295048952102661, 2.9958152770996094)),
    Vector((-11.986518859863281, -4.586348533630371, 2.9958152770996094)),
    Vector((-11.85675048828125, -4.898043632507324, 2.9958152770996094)),
    Vector((-12.06847858428955, -4.987098693847656, 2.9958152770996094)),
    Vector((-11.911391258239746, -5.3767170906066895, 2.9958152770996094)),
    Vector((-11.713323593139648, -5.298793792724609, 2.9958152770996094)),
    Vector((-10.142447471618652, -9.005736351013184, 2.9958152770996094)),
    Vector((-9.684842109680176, -8.805362701416016, 2.9958152770996094)),
    Vector((-9.54824447631836, -9.150452613830566, 2.9958152770996094)),
    Vector((-6.8231000900268555, -7.981602668762207, 2.9958152770996094)),
    Vector((-6.966527938842773, -7.636512279510498, 2.9958152770996094)),
    Vector((1.564052700996399, -4.007501602172852, 2.9958152770996094)),
    Vector((-0.10927870124578476, -0.044527795165777206, 2.9958152770996094))
]
unitVectors = [
    Vector((-0.37043139338493347, 0.9288597702980042, 0.0)),
    Vector((-0.9193736910820007, -0.3933852016925812, 0.0)),
    Vector((0.3206331431865692, -0.9472033977508545, 0.0)),
    Vector((-0.9211124777793884, -0.38929659128189087, 0.0)),
    Vector((-0.4770161509513855, 0.8788945078849792, 0.0)),
    Vector((-0.9200586080551147, -0.3917807936668396, 0.0)),
    Vector((0.3843514323234558, -0.9231868982315063, 0.0)),
    Vector((-0.921781063079834, -0.3877108097076416, 0.0)),
    Vector((0.3739338219165802, -0.9274554252624512, 0.0)),
    Vector((0.9305739402770996, 0.3661041259765625, 0.0)),
    Vector((0.39017802476882935, -0.9207394123077393, 0.0)),
    Vector((0.9160313606262207, 0.40110665559768677, 0.0)),
    Vector((0.3680473566055298, -0.9298070073127747, 0.0)),
    Vector((0.9190313220024109, 0.39418449997901917, 0.0)),
    Vector((-0.3837948441505432, 0.923418402671814, 0.0)),
    Vector((0.9201943874359131, 0.39146167039871216, 0.0)),
    Vector((-0.3889869749546051, 0.9212432503700256, 0.0)),
    Vector((0.9260721206665039, 0.37734663486480713, 0.0))
]
holesInfo = None
firstVertIndex = 18
numPolygonVerts = 18
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