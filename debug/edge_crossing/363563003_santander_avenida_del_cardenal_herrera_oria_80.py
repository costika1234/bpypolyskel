from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((5.268435955047607, 0.5343356132507324, 0.0)),
    Vector((5.389643669128418, -0.656782865524292, 0.0)),
    Vector((7.175417423248291, -0.47867000102996826, 0.0)),
    Vector((7.143095016479492, -0.0779198557138443, 0.0)),
    Vector((16.677993774414062, 0.8905766010284424, 0.0)),
    Vector((16.710315704345703, 0.489826500415802, 0.0)),
    Vector((18.512250900268555, 0.6790743470191956, 0.0)),
    Vector((18.382959365844727, 1.870192527770996, 0.0)),
    Vector((21.01717758178711, 2.137367010116577, 0.0)),
    Vector((20.209110260009766, 10.118971824645996, 0.0)),
    Vector((-34.93973159790039, 4.486266136169434, 0.0)),
    Vector((-34.13172912597656, -3.4842135906219482, 0.0)),
    Vector((-31.497509002685547, -3.21705961227417, 0.0)),
    Vector((-31.384387969970703, -4.419310569763184, 0.0)),
    Vector((-29.58245086669922, -4.230075836181641, 0.0)),
    Vector((-29.630931854248047, -3.8404572010040283, 0.0)),
    Vector((-20.08794403076172, -2.8608808517456055, 0.0)),
    Vector((-20.047542572021484, -3.2616312503814697, 0.0)),
    Vector((-18.26176643371582, -3.0723931789398193, 0.0)),
    Vector((-18.382970809936523, -1.8812743425369263, 0.0)),
    Vector((-13.114531517028809, -1.3358211517333984, 0.0)),
    Vector((-12.993327140808105, -2.538071870803833, 0.0)),
    Vector((-11.207551956176758, -2.34883189201355, 0.0)),
    Vector((-11.247953414916992, -1.9480817317962646, 0.0)),
    Vector((-1.713050127029419, -0.9796112775802612, 0.0)),
    Vector((-1.6645677089691162, -1.3803614377975464, 0.0)),
    Vector((0.1212063878774643, -1.2022504806518555, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((5.268435955047607, 0.5343356132507324, 14.151607513427734)),
    Vector((5.389643669128418, -0.656782865524292, 14.151607513427734)),
    Vector((7.175417423248291, -0.47867000102996826, 14.151607513427734)),
    Vector((7.143095016479492, -0.0779198557138443, 14.151607513427734)),
    Vector((16.677993774414062, 0.8905766010284424, 14.151607513427734)),
    Vector((16.710315704345703, 0.489826500415802, 14.151607513427734)),
    Vector((18.512250900268555, 0.6790743470191956, 14.151607513427734)),
    Vector((18.382959365844727, 1.870192527770996, 14.151607513427734)),
    Vector((21.01717758178711, 2.137367010116577, 14.151607513427734)),
    Vector((20.209110260009766, 10.118971824645996, 14.151607513427734)),
    Vector((-34.93973159790039, 4.486266136169434, 14.151607513427734)),
    Vector((-34.13172912597656, -3.4842135906219482, 14.151607513427734)),
    Vector((-31.497509002685547, -3.21705961227417, 14.151607513427734)),
    Vector((-31.384387969970703, -4.419310569763184, 14.151607513427734)),
    Vector((-29.58245086669922, -4.230075836181641, 14.151607513427734)),
    Vector((-29.630931854248047, -3.8404572010040283, 14.151607513427734)),
    Vector((-20.08794403076172, -2.8608808517456055, 14.151607513427734)),
    Vector((-20.047542572021484, -3.2616312503814697, 14.151607513427734)),
    Vector((-18.26176643371582, -3.0723931789398193, 14.151607513427734)),
    Vector((-18.382970809936523, -1.8812743425369263, 14.151607513427734)),
    Vector((-13.114531517028809, -1.3358211517333984, 14.151607513427734)),
    Vector((-12.993327140808105, -2.538071870803833, 14.151607513427734)),
    Vector((-11.207551956176758, -2.34883189201355, 14.151607513427734)),
    Vector((-11.247953414916992, -1.9480817317962646, 14.151607513427734)),
    Vector((-1.713050127029419, -0.9796112775802612, 14.151607513427734)),
    Vector((-1.6645677089691162, -1.3803614377975464, 14.151607513427734)),
    Vector((0.1212063878774643, -1.2022504806518555, 14.151607513427734)),
    Vector((0.0, 0.0, 14.151607513427734))
]
unitVectors = [
    Vector((0.10123676806688309, -0.9948622584342957, 0.0)),
    Vector((0.9950627684593201, 0.09924744069576263, 0.0)),
    Vector((-0.08039369434118271, 0.9967631697654724, 0.0)),
    Vector((0.9948809742927551, 0.10105389356613159, 0.0)),
    Vector((0.08039253205060959, -0.9967633485794067, 0.0)),
    Vector((0.9945300817489624, 0.1044503003358841, 0.0)),
    Vector((-0.10791248083114624, 0.9941604137420654, 0.0)),
    Vector((0.9948959350585938, 0.10090690851211548, 0.0)),
    Vector((-0.10072630643844604, 0.9949141144752502, 0.0)),
    Vector((-0.9948244690895081, -0.10160781443119049, 0.0)),
    Vector((0.10085746645927429, -0.9949009418487549, 0.0)),
    Vector((0.994896650314331, 0.10089915990829468, 0.0)),
    Vector((0.09367726743221283, -0.995602548122406, 0.0)),
    Vector((0.9945307970046997, 0.1044430285692215, 0.0)),
    Vector((-0.12347964197397232, 0.9923471212387085, 0.0)),
    Vector((0.9947729110717773, 0.10211225599050522, 0.0)),
    Vector((0.10030607134103775, -0.9949566125869751, 0.0)),
    Vector((0.9944321513175964, 0.1053796261548996, 0.0)),
    Vector((-0.10123398154973984, 0.9948625564575195, 0.0)),
    Vector((0.9946832656860352, 0.10298176109790802, 0.0)),
    Vector((0.10030611604452133, -0.9949566721916199, 0.0)),
    Vector((0.9944319128990173, 0.10538072139024734, 0.0)),
    Vector((-0.10030613094568253, 0.9949566125869751, 0.0)),
    Vector((0.9948812127113342, 0.10105115920305252, 0.0)),
    Vector((0.12010343372821808, -0.9927613139152527, 0.0)),
    Vector((0.9950628280639648, 0.09924636781215668, 0.0)),
    Vector((-0.10030778497457504, 0.9949564933776855, 0.0)),
    Vector((0.9948961138725281, 0.10090440511703491, 0.0))
]
holesInfo = None
firstVertIndex = 28
numPolygonVerts = 28

bpypolyskel.debugOutputs["skeleton"] = 1


faces = bpypolyskel.polygonize(verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, unitVectors)


# the number of vertices in a face
for face in faces:
    assert len(face) >= 3


# duplications of vertex indices
for face in faces:
    assert len(face) == len(set(face))


# edge crossing
assert not bpypolyskel.check_edge_crossing(bpypolyskel.debugOutputs["skeleton"])
