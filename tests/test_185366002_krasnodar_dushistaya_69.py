import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((6.572003364562988, -0.011128546670079231, 0.0)),
    Vector((6.572001934051514, 1.1911219358444214, 0.0)),
    Vector((9.061039924621582, 1.1799930334091187, 0.0)),
    Vector((9.061038970947266, 1.7031946182250977, 0.0)),
    Vector((9.924742698669434, 1.6920640468597412, 0.0)),
    Vector((9.932585716247559, 7.6810526847839355, 0.0)),
    Vector((10.568584442138672, 7.681053638458252, 0.0)),
    Vector((10.576432228088379, 10.631020545959473, 0.0)),
    Vector((12.233170509338379, 10.631023406982422, 0.0)),
    Vector((12.241015434265137, 14.104190826416016, 0.0)),
    Vector((13.253902435302734, 14.104193687438965, 0.0)),
    Vector((13.25389575958252, 17.33245849609375, 0.0)),
    Vector((10.961160659790039, 17.332454681396484, 0.0)),
    Vector((10.976852416992188, 23.889171600341797, 0.0)),
    Vector((12.437292098999023, 23.889175415039062, 0.0)),
    Vector((12.445137023925781, 27.44026756286621, 0.0)),
    Vector((11.000402450561523, 27.440263748168945, 0.0)),
    Vector((11.008247375488281, 30.946828842163086, 0.0)),
    Vector((10.435065269470215, 30.946826934814453, 0.0)),
    Vector((10.442907333374023, 36.835628509521484, 0.0)),
    Vector((6.681890487670898, 36.84675598144531, 0.0)),
    Vector((6.681889057159424, 37.94881820678711, 0.0)),
    Vector((0.0, 37.95994567871094, 0.0)),
    Vector((0.0, 36.77996063232422, 0.0)),
    Vector((-3.3762784004211426, 36.79109191894531, 0.0)),
    Vector((-3.3841333389282227, 30.835500717163086, 0.0)),
    Vector((-4.287092208862305, 30.835500717163086, 0.0)),
    Vector((-4.294946670532227, 27.306673049926758, 0.0)),
    Vector((-5.386349678039551, 27.306673049926758, 0.0)),
    Vector((-5.3863525390625, 24.022748947143555, 0.0)),
    Vector((-4.287096977233887, 24.022747039794922, 0.0)),
    Vector((-4.29495096206665, 20.794483184814453, 0.0)),
    Vector((-5.441318035125732, 20.794483184814453, 0.0)),
    Vector((-5.441320896148682, 17.410369873046875, 0.0)),
    Vector((-4.271397590637207, 17.410369873046875, 0.0)),
    Vector((-4.271399974822998, 14.104181289672852, 0.0)),
    Vector((-5.402064800262451, 14.104182243347168, 0.0)),
    Vector((-5.409919261932373, 10.664409637451172, 0.0)),
    Vector((-4.271402359008789, 10.664408683776855, 0.0)),
    Vector((-4.27140474319458, 7.213504314422607, 0.0)),
    Vector((-3.509775400161743, 7.213503837585449, 0.0)),
    Vector((-3.5176305770874023, 1.3358348608016968, 0.0)),
    Vector((0.0, 1.3358339071273804, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((6.572003364562988, -0.011128546670079231, 15.822172164916992)),
    Vector((6.572001934051514, 1.1911219358444214, 15.822172164916992)),
    Vector((9.061039924621582, 1.1799930334091187, 15.822172164916992)),
    Vector((9.061038970947266, 1.7031946182250977, 15.822172164916992)),
    Vector((9.924742698669434, 1.6920640468597412, 15.822172164916992)),
    Vector((9.932585716247559, 7.6810526847839355, 15.822172164916992)),
    Vector((10.568584442138672, 7.681053638458252, 15.822172164916992)),
    Vector((10.576432228088379, 10.631020545959473, 15.822172164916992)),
    Vector((12.233170509338379, 10.631023406982422, 15.822172164916992)),
    Vector((12.241015434265137, 14.104190826416016, 15.822172164916992)),
    Vector((13.253902435302734, 14.104193687438965, 15.822172164916992)),
    Vector((13.25389575958252, 17.33245849609375, 15.822172164916992)),
    Vector((10.961160659790039, 17.332454681396484, 15.822172164916992)),
    Vector((10.976852416992188, 23.889171600341797, 15.822172164916992)),
    Vector((12.437292098999023, 23.889175415039062, 15.822172164916992)),
    Vector((12.445137023925781, 27.44026756286621, 15.822172164916992)),
    Vector((11.000402450561523, 27.440263748168945, 15.822172164916992)),
    Vector((11.008247375488281, 30.946828842163086, 15.822172164916992)),
    Vector((10.435065269470215, 30.946826934814453, 15.822172164916992)),
    Vector((10.442907333374023, 36.835628509521484, 15.822172164916992)),
    Vector((6.681890487670898, 36.84675598144531, 15.822172164916992)),
    Vector((6.681889057159424, 37.94881820678711, 15.822172164916992)),
    Vector((0.0, 37.95994567871094, 15.822172164916992)),
    Vector((0.0, 36.77996063232422, 15.822172164916992)),
    Vector((-3.3762784004211426, 36.79109191894531, 15.822172164916992)),
    Vector((-3.3841333389282227, 30.835500717163086, 15.822172164916992)),
    Vector((-4.287092208862305, 30.835500717163086, 15.822172164916992)),
    Vector((-4.294946670532227, 27.306673049926758, 15.822172164916992)),
    Vector((-5.386349678039551, 27.306673049926758, 15.822172164916992)),
    Vector((-5.3863525390625, 24.022748947143555, 15.822172164916992)),
    Vector((-4.287096977233887, 24.022747039794922, 15.822172164916992)),
    Vector((-4.29495096206665, 20.794483184814453, 15.822172164916992)),
    Vector((-5.441318035125732, 20.794483184814453, 15.822172164916992)),
    Vector((-5.441320896148682, 17.410369873046875, 15.822172164916992)),
    Vector((-4.271397590637207, 17.410369873046875, 15.822172164916992)),
    Vector((-4.271399974822998, 14.104181289672852, 15.822172164916992)),
    Vector((-5.402064800262451, 14.104182243347168, 15.822172164916992)),
    Vector((-5.409919261932373, 10.664409637451172, 15.822172164916992)),
    Vector((-4.271402359008789, 10.664408683776855, 15.822172164916992)),
    Vector((-4.27140474319458, 7.213504314422607, 15.822172164916992)),
    Vector((-3.509775400161743, 7.213503837585449, 15.822172164916992)),
    Vector((-3.5176305770874023, 1.3358348608016968, 15.822172164916992)),
    Vector((0.0, 1.3358339071273804, 15.822172164916992)),
    Vector((0.0, 0.0, 15.822172164916992))
]
unitVectors = [
    Vector((-1.1898614502570126e-06, 1.0, 0.0)),
    Vector((0.9999900460243225, -0.004471121821552515, 0.0)),
    Vector((-1.8227665350423194e-06, 1.0, 0.0)),
    Vector((0.9999169707298279, -0.012885955162346363, 0.0)),
    Vector((0.0013095717877149582, 0.9999991655349731, 0.0)),
    Vector((1.0, 1.4994909633969655e-06, 0.0)),
    Vector((0.002660286845639348, 0.9999964833259583, 0.0)),
    Vector((0.9999999403953552, 1.7269009049414308e-06, 0.0)),
    Vector((0.002258717780932784, 0.9999974370002747, 0.0)),
    Vector((1.0, 2.8246220153960166e-06, 0.0)),
    Vector((-2.0678974124166416e-06, 1.0, 0.0)),
    Vector((-1.0, -1.6638194892948377e-06, 0.0)),
    Vector((0.002393227070569992, 0.9999971389770508, 0.0)),
    Vector((1.0, 2.612019670777954e-06, 0.0)),
    Vector((0.002209153026342392, 0.999997615814209, 0.0)),
    Vector((-1.0, -2.6404138679936295e-06, 0.0)),
    Vector((0.0022372049279510975, 0.9999974370002747, 0.0)),
    Vector((-1.0, -3.327648528284044e-06, 0.0)),
    Vector((0.001331689883954823, 0.9999991655349731, 0.0)),
    Vector((-0.9999955892562866, 0.002958620898425579, 0.0)),
    Vector((-1.2980315204913495e-06, 1.0, 0.0)),
    Vector((-0.9999986290931702, 0.0016653159400448203, 0.0)),
    Vector((0.0, -1.0, 0.0)),
    Vector((-0.9999945163726807, 0.0032968921586871147, 0.0)),
    Vector((-0.0013189171440899372, -0.9999991059303284, 0.0)),
    Vector((-1.0, 0.0, 0.0)),
    Vector((-0.0022257938981056213, -0.999997615814209, 0.0)),
    Vector((-1.0, 0.0, 0.0)),
    Vector((-8.712207772987313e-07, -0.9999999403953552, 0.0)),
    Vector((1.0, -1.73512762557948e-06, 0.0)),
    Vector((-0.0024328746367245913, -0.9999970197677612, 0.0)),
    Vector((-1.0, 0.0, 0.0)),
    Vector((-8.454276212432887e-07, -1.0, 0.0)),
    Vector((1.0, 0.0, 0.0)),
    Vector((-7.211281740637787e-07, -1.0, 0.0)),
    Vector((-1.0, 8.434633400611347e-07, 0.0)),
    Vector((-0.0022834185510873795, -0.9999973177909851, 0.0)),
    Vector((1.0, -8.376461551051761e-07, 0.0)),
    Vector((-6.908871910127345e-07, -0.9999999403953552, 0.0)),
    Vector((0.9999999403953552, -6.260750637920864e-07, 0.0)),
    Vector((-0.001336443005129695, -0.9999990463256836, 0.0)),
    Vector((1.0, -2.711126967369637e-07, 0.0)),
    Vector((0.0, -1.0, 0.0)),
    Vector((0.9999985098838806, -0.0016933238366618752, 0.0))
]
holesInfo = None
firstVertIndex = 44
numPolygonVerts = 44
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
    assert not bpypolyskel.check_edge_crossing(bpypolyskel.debugOutputs["skeleton"])
