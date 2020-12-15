import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-2.3390867710113525, -3.1058132648468018, 0.0)),
    Vector((-2.8747034072875977, -2.7273266315460205, 0.0)),
    Vector((-4.135776519775391, -3.2505273818969727, 0.0)),
    Vector((-5.539230823516846, -5.22088098526001, 0.0)),
    Vector((-5.390073299407959, -6.8016180992126465, 0.0)),
    Vector((-6.657929420471191, -8.404617309570312, 0.0)),
    Vector((-7.478305816650391, -8.460275650024414, 0.0)),
    Vector((-8.285123825073242, -9.37309455871582, 0.0)),
    Vector((-7.857988357543945, -11.23213005065918, 0.0)),
    Vector((-6.847773551940918, -11.866652488708496, 0.0)),
    Vector((-6.034176349639893, -11.566091537475586, 0.0)),
    Vector((-4.2849440574646, -12.835135459899902, 0.0)),
    Vector((-5.532461643218994, -14.482662200927734, 0.0)),
    Vector((-0.5762984752655029, -18.13394546508789, 0.0)),
    Vector((0.6915579438209534, -16.453020095825195, 0.0)),
    Vector((4.881588459014893, -19.492040634155273, 0.0)),
    Vector((3.640852689743042, -21.10617446899414, 0.0)),
    Vector((8.447869300842285, -24.534809112548828, 0.0)),
    Vector((9.553004264831543, -22.87614631652832, 0.0)),
    Vector((10.468304634094238, -23.5440616607666, 0.0)),
    Vector((10.414066314697266, -24.490276336669922, 0.0)),
    Vector((12.088726997375488, -25.636863708496094, 0.0)),
    Vector((14.054924011230469, -25.2249755859375, 0.0)),
    Vector((15.248196601867676, -23.35480499267578, 0.0)),
    Vector((14.868515014648438, -22.731416702270508, 0.0)),
    Vector((15.695670127868652, -21.595956802368164, 0.0)),
    Vector((16.136369705200195, -21.907649993896484, 0.0)),
    Vector((17.31608772277832, -21.785194396972656, 0.0)),
    Vector((19.01785659790039, -19.569929122924805, 0.0)),
    Vector((18.61104965209961, -17.777687072753906, 0.0)),
    Vector((20.787412643432617, -14.894503593444824, 0.0)),
    Vector((0.0, 0.0, 10.736668586730957)),
    Vector((-2.3390867710113525, -3.1058132648468018, 10.736668586730957)),
    Vector((-2.8747034072875977, -2.7273266315460205, 10.736668586730957)),
    Vector((-4.135776519775391, -3.2505273818969727, 10.736668586730957)),
    Vector((-5.539230823516846, -5.22088098526001, 10.736668586730957)),
    Vector((-5.390073299407959, -6.8016180992126465, 10.736668586730957)),
    Vector((-6.657929420471191, -8.404617309570312, 10.736668586730957)),
    Vector((-7.478305816650391, -8.460275650024414, 10.736668586730957)),
    Vector((-8.285123825073242, -9.37309455871582, 10.736668586730957)),
    Vector((-7.857988357543945, -11.23213005065918, 10.736668586730957)),
    Vector((-6.847773551940918, -11.866652488708496, 10.736668586730957)),
    Vector((-6.034176349639893, -11.566091537475586, 10.736668586730957)),
    Vector((-4.2849440574646, -12.835135459899902, 10.736668586730957)),
    Vector((-5.532461643218994, -14.482662200927734, 10.736668586730957)),
    Vector((-0.5762984752655029, -18.13394546508789, 10.736668586730957)),
    Vector((0.6915579438209534, -16.453020095825195, 10.736668586730957)),
    Vector((4.881588459014893, -19.492040634155273, 10.736668586730957)),
    Vector((3.640852689743042, -21.10617446899414, 10.736668586730957)),
    Vector((8.447869300842285, -24.534809112548828, 10.736668586730957)),
    Vector((9.553004264831543, -22.87614631652832, 10.736668586730957)),
    Vector((10.468304634094238, -23.5440616607666, 10.736668586730957)),
    Vector((10.414066314697266, -24.490276336669922, 10.736668586730957)),
    Vector((12.088726997375488, -25.636863708496094, 10.736668586730957)),
    Vector((14.054924011230469, -25.2249755859375, 10.736668586730957)),
    Vector((15.248196601867676, -23.35480499267578, 10.736668586730957)),
    Vector((14.868515014648438, -22.731416702270508, 10.736668586730957)),
    Vector((15.695670127868652, -21.595956802368164, 10.736668586730957)),
    Vector((16.136369705200195, -21.907649993896484, 10.736668586730957)),
    Vector((17.31608772277832, -21.785194396972656, 10.736668586730957)),
    Vector((19.01785659790039, -19.569929122924805, 10.736668586730957)),
    Vector((18.61104965209961, -17.777687072753906, 10.736668586730957)),
    Vector((20.787412643432617, -14.894503593444824, 10.736668586730957))
]
unitVectors = [
    Vector((-0.6015998721122742, -0.798797607421875, 0.0)),
    Vector((-0.8166773319244385, 0.577094554901123, 0.0)),
    Vector((-0.9236599206924438, -0.38321295380592346, 0.0)),
    Vector((-0.5801589488983154, -0.8145033717155457, 0.0)),
    Vector((0.09394218772649765, -0.9955776929855347, 0.0)),
    Vector((-0.6203464865684509, -0.7843279242515564, 0.0)),
    Vector((-0.9977064728736877, -0.06768927723169327, 0.0)),
    Vector((-0.6622627377510071, -0.7492717504501343, 0.0)),
    Vector((0.22392725944519043, -0.9746058583259583, 0.0)),
    Vector((0.8468138575553894, -0.5318892598152161, 0.0)),
    Vector((0.9380381107330322, 0.3465321958065033, 0.0)),
    Vector((0.8094233870506287, -0.5872254967689514, 0.0)),
    Vector((-0.6036704778671265, -0.7972338795661926, 0.0)),
    Vector((0.8051044344902039, -0.5931330919265747, 0.0)),
    Vector((0.6021749377250671, 0.7983642220497131, 0.0)),
    Vector((0.809495747089386, -0.5871255993843079, 0.0)),
    Vector((-0.609431266784668, -0.7928389310836792, 0.0)),
    Vector((0.814129650592804, -0.5806831121444702, 0.0)),
    Vector((0.5544778108596802, 0.8321985602378845, 0.0)),
    Vector((0.8077936768531799, -0.589465320110321, 0.0)),
    Vector((-0.057227425277233124, -0.9983611106872559, 0.0)),
    Vector((0.8251311182975769, -0.5649412870407104, 0.0)),
    Vector((0.9787548780441284, 0.20503413677215576, 0.0)),
    Vector((0.5378902554512024, 0.8430148959159851, 0.0)),
    Vector((-0.5201746821403503, 0.8540598750114441, 0.0)),
    Vector((0.5888075232505798, 0.8082732558250427, 0.0)),
    Vector((0.8164340853691101, -0.5774385929107666, 0.0)),
    Vector((0.9946558475494385, 0.10324601083993912, 0.0)),
    Vector((0.6091975569725037, 0.7930185198783875, 0.0)),
    Vector((-0.22135165333747864, 0.9751941561698914, 0.0)),
    Vector((0.6024731397628784, 0.798139214515686, 0.0)),
    Vector((-0.8128753304481506, 0.5824377536773682, 0.0))
]
holesInfo = None
firstVertIndex = 32
numPolygonVerts = 32
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