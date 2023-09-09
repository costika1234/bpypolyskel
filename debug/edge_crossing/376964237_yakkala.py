from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.011046336963772774, -0.20037508010864258, 0.0)),
    Vector((12.968398094177246, 0.6567866206169128, 0.0)),
    Vector((12.74747085571289, 4.308065891265869, 0.0)),
    Vector((12.548636436462402, 4.296933650970459, 0.0)),
    Vector((12.195152282714844, 9.90743637084961, 0.0)),
    Vector((7.301626682281494, 9.60687255859375, 0.0)),
    Vector((7.6551103591918945, 3.9629745483398438, 0.0)),
    Vector((4.617368221282959, 3.773730993270874, 0.0)),
    Vector((4.308070182800293, 8.682920455932617, 0.0)),
    Vector((-0.3866216838359833, 8.393489837646484, 0.0)),
    Vector((-0.07732434570789337, 3.450904130935669, 0.0)),
    Vector((-0.24301937222480774, 3.439772367477417, 0.0)),
    Vector((-0.03313900902867317, -0.011131948791444302, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((0.011046336963772774, -0.20037508010864258, 2.884079694747925)),
    Vector((12.968398094177246, 0.6567866206169128, 2.884079694747925)),
    Vector((12.74747085571289, 4.308065891265869, 2.884079694747925)),
    Vector((12.548636436462402, 4.296933650970459, 2.884079694747925)),
    Vector((12.195152282714844, 9.90743637084961, 2.884079694747925)),
    Vector((7.301626682281494, 9.60687255859375, 2.884079694747925)),
    Vector((7.6551103591918945, 3.9629745483398438, 2.884079694747925)),
    Vector((4.617368221282959, 3.773730993270874, 2.884079694747925)),
    Vector((4.308070182800293, 8.682920455932617, 2.884079694747925)),
    Vector((-0.3866216838359833, 8.393489837646484, 2.884079694747925)),
    Vector((-0.07732434570789337, 3.450904130935669, 2.884079694747925)),
    Vector((-0.24301937222480774, 3.439772367477417, 2.884079694747925)),
    Vector((-0.03313900902867317, -0.011131948791444302, 2.884079694747925)),
    Vector((0.0, 0.0, 2.884079694747925))
]
unitVectors = [
    Vector((0.9978190660476685, 0.06600826233625412, 0.0)),
    Vector((-0.06039634719491005, 0.9981744289398193, 0.0)),
    Vector((-0.9984363317489624, -0.05589994788169861, 0.0)),
    Vector((-0.06287932395935059, 0.9980210661888123, 0.0)),
    Vector((-0.998119056224823, -0.06130518019199371, 0.0)),
    Vector((0.06250864267349243, -0.9980444312095642, 0.0)),
    Vector((-0.998065173625946, -0.0621769018471241, 0.0)),
    Vector((-0.06287921220064163, 0.9980210661888123, 0.0)),
    Vector((-0.998104989528656, -0.06153378263115883, 0.0)),
    Vector((0.06245587021112442, -0.9980477690696716, 0.0)),
    Vector((-0.9977508783340454, -0.0670311376452446, 0.0)),
    Vector((0.06070677191019058, -0.9981556534767151, 0.0)),
    Vector((0.9479460716247559, 0.31843098998069763, 0.0)),
    Vector((0.05504471808671951, -0.9984838962554932, 0.0))
]
holesInfo = None
firstVertIndex = 14
numPolygonVerts = 14

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
