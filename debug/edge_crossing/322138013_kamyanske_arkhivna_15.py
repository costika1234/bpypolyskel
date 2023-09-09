from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((46.63177490234375, 2.003943681716919, 0.0)),
    Vector((46.20408248901367, 11.844583511352539, 0.0)),
    Vector((37.92465591430664, 11.499430656433105, 0.0)),
    Vector((37.5854606628418, 19.46990394592285, 0.0)),
    Vector((27.426034927368164, 19.03569984436035, 0.0)),
    Vector((27.765213012695312, 11.054093360900879, 0.0)),
    Vector((18.33565902709961, 10.653305053710938, 0.0)),
    Vector((17.99649429321289, 18.64604377746582, 0.0)),
    Vector((7.8296918869018555, 18.200742721557617, 0.0)),
    Vector((8.176215171813965, 10.219135284423828, 0.0)),
    Vector((-0.42023831605911255, 9.851775169372559, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((46.63177490234375, 2.003943681716919, 8.519157409667969)),
    Vector((46.20408248901367, 11.844583511352539, 8.519157409667969)),
    Vector((37.92465591430664, 11.499430656433105, 8.519157409667969)),
    Vector((37.5854606628418, 19.46990394592285, 8.519157409667969)),
    Vector((27.426034927368164, 19.03569984436035, 8.519157409667969)),
    Vector((27.765213012695312, 11.054093360900879, 8.519157409667969)),
    Vector((18.33565902709961, 10.653305053710938, 8.519157409667969)),
    Vector((17.99649429321289, 18.64604377746582, 8.519157409667969)),
    Vector((7.8296918869018555, 18.200742721557617, 8.519157409667969)),
    Vector((8.176215171813965, 10.219135284423828, 8.519157409667969)),
    Vector((-0.42023831605911255, 9.851775169372559, 8.519157409667969)),
    Vector((0.0, 0.0, 8.519157409667969))
]
unitVectors = [
    Vector((-0.043420858681201935, 0.9990568161010742, 0.0)),
    Vector((-0.9991320967674255, -0.041651833802461624, 0.0)),
    Vector((-0.04251798614859581, 0.999095618724823, 0.0)),
    Vector((-0.9990879893302917, -0.042700059711933136, 0.0)),
    Vector((0.042456645518541336, -0.9990983009338379, 0.0)),
    Vector((-0.9990979433059692, -0.04246508330106735, 0.0)),
    Vector((-0.04239595681428909, 0.999100923538208, 0.0)),
    Vector((-0.9990420937538147, -0.043757565319538116, 0.0)),
    Vector((0.04337436705827713, -0.9990589022636414, 0.0)),
    Vector((-0.9990881681442261, -0.042694948613643646, 0.0)),
    Vector((0.04261734336614609, -0.9990914463996887, 0.0)),
    Vector((0.9990779161453247, 0.04293415695428848, 0.0))
]
holesInfo = None
firstVertIndex = 12
numPolygonVerts = 12

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
