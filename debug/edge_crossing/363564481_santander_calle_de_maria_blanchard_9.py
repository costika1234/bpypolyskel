from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.12120599299669266, -1.191118597984314, 0.0)),
    Vector((1.906974196434021, -1.0130070447921753, 0.0)),
    Vector((1.8746525049209595, -0.6122569441795349, 0.0)),
    Vector((11.417601585388184, 0.3451001048088074, 0.0)),
    Vector((11.44992446899414, -0.0556500069797039, 0.0)),
    Vector((13.24377155303955, 0.12246447056531906, 0.0)),
    Vector((13.122563362121582, 1.3247147798538208, 0.0)),
    Vector((18.390979766845703, 1.8479286432266235, 0.0)),
    Vector((18.50411033630371, 0.6568104028701782, 0.0)),
    Vector((20.297956466674805, 0.8349267840385437, 0.0)),
    Vector((20.257553100585938, 1.235676884651184, 0.0)),
    Vector((29.792417526245117, 2.1930599212646484, 0.0)),
    Vector((29.84090232849121, 1.7923099994659424, 0.0)),
    Vector((31.62666893005371, 1.9815611839294434, 0.0)),
    Vector((31.505456924438477, 3.1726791858673096, 0.0)),
    Vector((36.77387237548828, 3.7070395946502686, 0.0)),
    Vector((36.895084381103516, 2.5159215927124023, 0.0)),
    Vector((38.680850982666016, 2.694042921066284, 0.0)),
    Vector((38.640445709228516, 3.0947928428649902, 0.0)),
    Vector((48.18338394165039, 4.063333988189697, 0.0)),
    Vector((48.2157096862793, 3.662583827972412, 0.0)),
    Vector((50.017635345458984, 3.8407082557678223, 0.0)),
    Vector((49.88833999633789, 5.031826019287109, 0.0)),
    Vector((52.522544860839844, 5.299012660980225, 0.0)),
    Vector((51.72252655029297, 13.280613899230957, 0.0)),
    Vector((-3.4341652393341064, 7.714441776275635, 0.0)),
    Vector((-2.6342098712921143, -0.26716625690460205, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((0.12120599299669266, -1.191118597984314, 15.667717933654785)),
    Vector((1.906974196434021, -1.0130070447921753, 15.667717933654785)),
    Vector((1.8746525049209595, -0.6122569441795349, 15.667717933654785)),
    Vector((11.417601585388184, 0.3451001048088074, 15.667717933654785)),
    Vector((11.44992446899414, -0.0556500069797039, 15.667717933654785)),
    Vector((13.24377155303955, 0.12246447056531906, 15.667717933654785)),
    Vector((13.122563362121582, 1.3247147798538208, 15.667717933654785)),
    Vector((18.390979766845703, 1.8479286432266235, 15.667717933654785)),
    Vector((18.50411033630371, 0.6568104028701782, 15.667717933654785)),
    Vector((20.297956466674805, 0.8349267840385437, 15.667717933654785)),
    Vector((20.257553100585938, 1.235676884651184, 15.667717933654785)),
    Vector((29.792417526245117, 2.1930599212646484, 15.667717933654785)),
    Vector((29.84090232849121, 1.7923099994659424, 15.667717933654785)),
    Vector((31.62666893005371, 1.9815611839294434, 15.667717933654785)),
    Vector((31.505456924438477, 3.1726791858673096, 15.667717933654785)),
    Vector((36.77387237548828, 3.7070395946502686, 15.667717933654785)),
    Vector((36.895084381103516, 2.5159215927124023, 15.667717933654785)),
    Vector((38.680850982666016, 2.694042921066284, 15.667717933654785)),
    Vector((38.640445709228516, 3.0947928428649902, 15.667717933654785)),
    Vector((48.18338394165039, 4.063333988189697, 15.667717933654785)),
    Vector((48.2157096862793, 3.662583827972412, 15.667717933654785)),
    Vector((50.017635345458984, 3.8407082557678223, 15.667717933654785)),
    Vector((49.88833999633789, 5.031826019287109, 15.667717933654785)),
    Vector((52.522544860839844, 5.299012660980225, 15.667717933654785)),
    Vector((51.72252655029297, 13.280613899230957, 15.667717933654785)),
    Vector((-3.4341652393341064, 7.714441776275635, 15.667717933654785)),
    Vector((-2.6342098712921143, -0.26716625690460205, 15.667717933654785)),
    Vector((0.0, 0.0, 15.667717933654785))
]
unitVectors = [
    Vector((0.9950628280639648, 0.0992470309138298, 0.0)),
    Vector((-0.08039193600416183, 0.9967633485794067, 0.0)),
    Vector((0.9950055480003357, 0.0998198390007019, 0.0)),
    Vector((0.08039487898349762, -0.9967630505561829, 0.0)),
    Vector((0.9951066970825195, 0.09880602359771729, 0.0)),
    Vector((-0.10030926018953323, 0.9949562549591064, 0.0)),
    Vector((0.9951047301292419, 0.09882525354623795, 0.0)),
    Vector((0.09455293416976929, -0.9955198168754578, 0.0)),
    Vector((0.9951066970825195, 0.09880713373422623, 0.0)),
    Vector((-0.10031083971261978, 0.9949561953544617, 0.0)),
    Vector((0.9949968457221985, 0.0999063104391098, 0.0)),
    Vector((0.12010932713747025, -0.9927606582641602, 0.0)),
    Vector((0.99443119764328, 0.10538738965988159, 0.0)),
    Vector((-0.1012403666973114, 0.9948620200157166, 0.0)),
    Vector((0.9948955774307251, 0.10090943425893784, 0.0)),
    Vector((0.1012403666973114, -0.9948620200157166, 0.0)),
    Vector((0.9950622916221619, 0.09925250709056854, 0.0)),
    Vector((-0.10031556338071823, 0.9949556589126587, 0.0)),
    Vector((0.994888961315155, 0.10097423940896988, 0.0)),
    Vector((0.0804019421339035, -0.9967625141143799, 0.0)),
    Vector((0.995149552822113, 0.09837278723716736, 0.0)),
    Vector((-0.10791567713022232, 0.9941601157188416, 0.0)),
    Vector((0.9948953986167908, 0.1009119525551796, 0.0)),
    Vector((-0.09973306953907013, 0.9950142502784729, 0.0)),
    Vector((-0.9949465990066528, -0.10040565580129623, 0.0)),
    Vector((0.09972521662712097, -0.9950149655342102, 0.0)),
    Vector((0.9948961734771729, 0.10090414434671402, 0.0)),
    Vector((0.10123534500598907, -0.9948624968528748, 0.0))
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
