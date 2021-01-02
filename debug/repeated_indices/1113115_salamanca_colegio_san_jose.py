import pytest
from mathutils import Vector
from bpypolyskel import bpypolyskel


verts = [
    Vector((86.88787841796875, -23.165071487426758, 0.0)),
    Vector((86.88768768310547, -7.613739013671875, 0.0)),
    Vector((71.17243957519531, -7.446928977966309, 0.0)),
    Vector((71.17247009277344, -10.786513328552246, 0.0)),
    Vector((58.96162796020508, -10.61964225769043, 0.0)),
    Vector((58.62545394897461, -7.947977542877197, 0.0)),
    Vector((49.431610107421875, -7.948045253753662, 0.0)),
    Vector((49.4316291809082, -10.78669261932373, 0.0)),
    Vector((37.052711486816406, -10.786765098571777, 0.0)),
    Vector((37.052696228027344, -7.948118209838867, 0.0)),
    Vector((24.85026741027832, -7.781190395355225, 0.0)),
    Vector((24.51410484313965, -5.944419860839844, 0.0)),
    Vector((21.665191650390625, -6.278387546539307, 0.0)),
    Vector((20.665119171142578, -2.259756565093994, 0.0)),
    Vector((19.328895568847656, 1.079824447631836, 0.0)),
    Vector((17.320363998413086, 3.7626192569732666, 0.0)),
    Vector((15.311837196350098, 5.7663655281066895, 0.0)),
    Vector((20.833154678344727, 12.62366008758545, 0.0)),
    Vector((16.32027816772461, 15.629274368286133, 0.0)),
    Vector((8.622347831726074, 18.646018981933594, 0.0)),
    Vector((19.66487693786621, 65.80097961425781, 0.0)),
    Vector((4.44560432434082, 69.81958770751953, 0.0)),
    Vector((-8.43746566772461, 16.130199432373047, 0.0)),
    Vector((-5.756638050079346, 15.462279319763184, 0.0)),
    Vector((-3.748120069503784, 10.619880676269531, 0.0)),
    Vector((-6.260874271392822, 9.272915840148926, 0.0)),
    Vector((-5.252412796020508, 7.269164562225342, 0.0)),
    Vector((-7.933246612548828, 5.4323954582214355, 0.0)),
    Vector((-11.277989387512207, 1.7477246522903442, 0.0)),
    Vector((-13.286520957946777, -2.426752805709839, 0.0)),
    Vector((-14.118510246276855, -5.6104888916015625, 0.0)),
    Vector((-14.622750282287598, -9.28403091430664, 0.0)),
    Vector((-13.614290237426758, -12.122679710388184, 0.0)),
    Vector((-15.622819900512695, -12.968704223632812, 0.0)),
    Vector((-14.790839195251465, -14.972456932067871, 0.0)),
    Vector((-12.614232063293457, -14.47152328491211, 0.0)),
    Vector((-10.605709075927734, -18.646007537841797, 0.0)),
    Vector((-8.269432067871094, -21.829748153686523, 0.0)),
    Vector((-4.084293842315674, -24.334440231323242, 0.0)),
    Vector((-5.756672382354736, -28.186092376708984, 0.0)),
    Vector((-2.916154146194458, -29.187969207763672, 0.0)),
    Vector((-2.445535182952881, -28.297414779663086, 0.0)),
    Vector((-0.26892486214637756, -29.087783813476562, 0.0)),
    Vector((2.3951122760772705, -29.27702522277832, 0.0)),
    Vector((2.4959592819213867, -30.367956161499023, 0.0)),
    Vector((5.4541335105896, -30.468143463134766, 0.0)),
    Vector((5.454130172729492, -26.12668228149414, 0.0)),
    Vector((8.118165969848633, -25.625741958618164, 0.0)),
    Vector((10.88304615020752, -24.545940399169922, 0.0)),
    Vector((13.25294303894043, -23.154441833496094, 0.0)),
    Vector((15.135420799255371, -25.72591781616211, 0.0)),
    Vector((17.00948715209961, -24.345552444458008, 0.0)),
    Vector((19.37938117980957, -22.275005340576172, 0.0)),
    Vector((21.555980682373047, -19.503143310546875, 0.0)),
    Vector((22.83336639404297, -17.43259620666504, 0.0)),
    Vector((27.774852752685547, -17.3323917388916, 0.0)),
    Vector((27.77487564086914, -23.555150985717773, 0.0)),
    Vector((37.05277633666992, -23.655298233032227, 0.0)),
    Vector((37.052791595458984, -26.51620864868164, 0.0)),
    Vector((49.99479675292969, -26.415945053100586, 0.0)),
    Vector((49.89392852783203, -23.354660034179688, 0.0)),
    Vector((58.785247802734375, -23.254405975341797, 0.0)),
    Vector((58.785274505615234, -26.315692901611328, 0.0)),
    Vector((71.32389068603516, -26.315580368041992, 0.0)),
    Vector((71.32386016845703, -23.354482650756836, 0.0)),
    Vector((86.88787841796875, -23.165071487426758, 8.263378143310547)),
    Vector((86.88768768310547, -7.613739013671875, 8.263378143310547)),
    Vector((71.17243957519531, -7.446928977966309, 8.263378143310547)),
    Vector((71.17247009277344, -10.786513328552246, 8.263378143310547)),
    Vector((58.96162796020508, -10.61964225769043, 8.263378143310547)),
    Vector((58.62545394897461, -7.947977542877197, 8.263378143310547)),
    Vector((49.431610107421875, -7.948045253753662, 8.263378143310547)),
    Vector((49.4316291809082, -10.78669261932373, 8.263378143310547)),
    Vector((37.052711486816406, -10.786765098571777, 8.263378143310547)),
    Vector((37.052696228027344, -7.948118209838867, 8.263378143310547)),
    Vector((24.85026741027832, -7.781190395355225, 8.263378143310547)),
    Vector((24.51410484313965, -5.944419860839844, 8.263378143310547)),
    Vector((21.665191650390625, -6.278387546539307, 8.263378143310547)),
    Vector((20.665119171142578, -2.259756565093994, 8.263378143310547)),
    Vector((19.328895568847656, 1.079824447631836, 8.263378143310547)),
    Vector((17.320363998413086, 3.7626192569732666, 8.263378143310547)),
    Vector((15.311837196350098, 5.7663655281066895, 8.263378143310547)),
    Vector((20.833154678344727, 12.62366008758545, 8.263378143310547)),
    Vector((16.32027816772461, 15.629274368286133, 8.263378143310547)),
    Vector((8.622347831726074, 18.646018981933594, 8.263378143310547)),
    Vector((19.66487693786621, 65.80097961425781, 8.263378143310547)),
    Vector((4.44560432434082, 69.81958770751953, 8.263378143310547)),
    Vector((-8.43746566772461, 16.130199432373047, 8.263378143310547)),
    Vector((-5.756638050079346, 15.462279319763184, 8.263378143310547)),
    Vector((-3.748120069503784, 10.619880676269531, 8.263378143310547)),
    Vector((-6.260874271392822, 9.272915840148926, 8.263378143310547)),
    Vector((-5.252412796020508, 7.269164562225342, 8.263378143310547)),
    Vector((-7.933246612548828, 5.4323954582214355, 8.263378143310547)),
    Vector((-11.277989387512207, 1.7477246522903442, 8.263378143310547)),
    Vector((-13.286520957946777, -2.426752805709839, 8.263378143310547)),
    Vector((-14.118510246276855, -5.6104888916015625, 8.263378143310547)),
    Vector((-14.622750282287598, -9.28403091430664, 8.263378143310547)),
    Vector((-13.614290237426758, -12.122679710388184, 8.263378143310547)),
    Vector((-15.622819900512695, -12.968704223632812, 8.263378143310547)),
    Vector((-14.790839195251465, -14.972456932067871, 8.263378143310547)),
    Vector((-12.614232063293457, -14.47152328491211, 8.263378143310547)),
    Vector((-10.605709075927734, -18.646007537841797, 8.263378143310547)),
    Vector((-8.269432067871094, -21.829748153686523, 8.263378143310547)),
    Vector((-4.084293842315674, -24.334440231323242, 8.263378143310547)),
    Vector((-5.756672382354736, -28.186092376708984, 8.263378143310547)),
    Vector((-2.916154146194458, -29.187969207763672, 8.263378143310547)),
    Vector((-2.445535182952881, -28.297414779663086, 8.263378143310547)),
    Vector((-0.26892486214637756, -29.087783813476562, 8.263378143310547)),
    Vector((2.3951122760772705, -29.27702522277832, 8.263378143310547)),
    Vector((2.4959592819213867, -30.367956161499023, 8.263378143310547)),
    Vector((5.4541335105896, -30.468143463134766, 8.263378143310547)),
    Vector((5.454130172729492, -26.12668228149414, 8.263378143310547)),
    Vector((8.118165969848633, -25.625741958618164, 8.263378143310547)),
    Vector((10.88304615020752, -24.545940399169922, 8.263378143310547)),
    Vector((13.25294303894043, -23.154441833496094, 8.263378143310547)),
    Vector((15.135420799255371, -25.72591781616211, 8.263378143310547)),
    Vector((17.00948715209961, -24.345552444458008, 8.263378143310547)),
    Vector((19.37938117980957, -22.275005340576172, 8.263378143310547)),
    Vector((21.555980682373047, -19.503143310546875, 8.263378143310547)),
    Vector((22.83336639404297, -17.43259620666504, 8.263378143310547)),
    Vector((27.774852752685547, -17.3323917388916, 8.263378143310547)),
    Vector((27.77487564086914, -23.555150985717773, 8.263378143310547)),
    Vector((37.05277633666992, -23.655298233032227, 8.263378143310547)),
    Vector((37.052791595458984, -26.51620864868164, 8.263378143310547)),
    Vector((49.99479675292969, -26.415945053100586, 8.263378143310547)),
    Vector((49.89392852783203, -23.354660034179688, 8.263378143310547)),
    Vector((58.785247802734375, -23.254405975341797, 8.263378143310547)),
    Vector((58.785274505615234, -26.315692901611328, 8.263378143310547)),
    Vector((71.32389068603516, -26.315580368041992, 8.263378143310547)),
    Vector((71.32386016845703, -23.354482650756836, 8.263378143310547)),
    Vector((0.0, 0.0, 0.0)),
    Vector((3.092623472213745, 0.946216344833374, 0.0)),
    Vector((8.210580825805664, -0.6790443062782288, 0.0)),
    Vector((11.034285545349121, -3.4954237937927246, 0.0)),
    Vector((12.244449615478516, -7.402735710144043, 0.0)),
    Vector((10.899836540222168, -12.51230239868164, 0.0)),
    Vector((8.613983154296875, -15.20623779296875, 0.0)),
    Vector((7.134897708892822, -13.324939727783203, 0.0)),
    Vector((5.386890411376953, -14.12644100189209, 0.0)),
    Vector((4.437249660491943, -11.031760215759277, 0.0)),
    Vector((1.4790830612182617, -10.497427940368652, 0.0)),
    Vector((0.5378482341766357, -8.33782958984375, 0.0)),
    Vector((2.016930103302002, -6.055779933929443, 0.0)),
    Vector((-0.26892393827438354, -3.6290154457092285, 0.0)),
    Vector((1.2101575136184692, -2.426764726638794, 0.0)),
    Vector((0.0, 0.0, 8.263378143310547)),
    Vector((3.092623472213745, 0.946216344833374, 8.263378143310547)),
    Vector((8.210580825805664, -0.6790443062782288, 8.263378143310547)),
    Vector((11.034285545349121, -3.4954237937927246, 8.263378143310547)),
    Vector((12.244449615478516, -7.402735710144043, 8.263378143310547)),
    Vector((10.899836540222168, -12.51230239868164, 8.263378143310547)),
    Vector((8.613983154296875, -15.20623779296875, 8.263378143310547)),
    Vector((7.134897708892822, -13.324939727783203, 8.263378143310547)),
    Vector((5.386890411376953, -14.12644100189209, 8.263378143310547)),
    Vector((4.437249660491943, -11.031760215759277, 8.263378143310547)),
    Vector((1.4790830612182617, -10.497427940368652, 8.263378143310547)),
    Vector((0.5378482341766357, -8.33782958984375, 8.263378143310547)),
    Vector((2.016930103302002, -6.055779933929443, 8.263378143310547)),
    Vector((-0.26892393827438354, -3.6290154457092285, 8.263378143310547)),
    Vector((1.2101575136184692, -2.426764726638794, 8.263378143310547)),
    Vector((2.7396538257598877, 20.22675132751465, 0.0)),
    Vector((-1.2185574769973755, 21.19523048400879, 0.0)),
    Vector((-0.02521151676774025, 26.04876136779785, 0.0)),
    Vector((3.932997226715088, 25.08028221130371, 0.0)),
    Vector((2.7396538257598877, 20.22675132751465, 8.263378143310547)),
    Vector((-1.2185574769973755, 21.19523048400879, 8.263378143310547)),
    Vector((-0.02521151676774025, 26.04876136779785, 8.263378143310547)),
    Vector((3.932997226715088, 25.08028221130371, 8.263378143310547)),
    Vector((5.445683479309082, 31.88190460205078, 0.0)),
    Vector((1.4958819150924683, 32.85038375854492, 0.0)),
    Vector((2.6892242431640625, 37.70391082763672, 0.0)),
    Vector((6.639023303985596, 36.735435485839844, 0.0)),
    Vector((5.445683479309082, 31.88190460205078, 8.263378143310547)),
    Vector((1.4958819150924683, 32.85038375854492, 8.263378143310547)),
    Vector((2.6892242431640625, 37.70391082763672, 8.263378143310547)),
    Vector((6.639023303985596, 36.735435485839844, 8.263378143310547)),
    Vector((8.429030418395996, 43.92667770385742, 0.0)),
    Vector((4.470830917358398, 44.90628433227539, 0.0)),
    Vector((5.6641693115234375, 49.75981521606445, 0.0)),
    Vector((9.622365951538086, 48.780208587646484, 0.0)),
    Vector((8.429030418395996, 43.92667770385742, 8.263378143310547)),
    Vector((4.470830917358398, 44.90628433227539, 8.263378143310547)),
    Vector((5.6641693115234375, 49.75981521606445, 8.263378143310547)),
    Vector((9.622365951538086, 48.780208587646484, 8.263378143310547)),
    Vector((11.336732864379883, 55.860130310058594, 0.0)),
    Vector((7.386943817138672, 56.82860565185547, 0.0)),
    Vector((8.580278396606445, 61.68213653564453, 0.0)),
    Vector((12.530064582824707, 60.713661193847656, 0.0)),
    Vector((11.336732864379883, 55.860130310058594, 8.263378143310547)),
    Vector((7.386943817138672, 56.82860565185547, 8.263378143310547)),
    Vector((8.580278396606445, 61.68213653564453, 8.263378143310547)),
    Vector((12.530064582824707, 60.713661193847656, 8.263378143310547))
]
unitVectors = [
    Vector((-1.2264857105037663e-05, 1.0, 0.0)),
    Vector((-0.9999436736106873, 0.010613935999572277, 0.0)),
    Vector((9.138136192632373e-06, -0.9999999403953552, 0.0)),
    Vector((-0.9999066591262817, 0.01366453617811203, 0.0)),
    Vector((-0.12484496831893921, 0.992176353931427, 0.0)),
    Vector((-1.0, -7.364806151599623e-06, 0.0)),
    Vector((6.71921679895604e-06, -1.0, 0.0)),
    Vector((-0.9999999403953552, -5.855054951098282e-06, 0.0)),
    Vector((-5.375374257710064e-06, 1.0, 0.0)),
    Vector((-0.9999064207077026, 0.013678603805601597, 0.0)),
    Vector((-0.180028036236763, 0.9836615324020386, 0.0)),
    Vector((-0.9931989908218384, -0.11642909049987793, 0.0)),
    Vector((-0.24149340391159058, 0.9704025387763977, 0.0)),
    Vector((-0.3714844286441803, 0.9284391403198242, 0.0)),
    Vector((-0.5993191003799438, 0.8005102872848511, 0.0)),
    Vector((-0.7079488039016724, 0.7062638401985168, 0.0)),
    Vector((0.6271494030952454, 0.7788988947868347, 0.0)),
    Vector((-0.832302987575531, 0.5543208718299866, 0.0)),
    Vector((-0.9310574531555176, 0.36487242579460144, 0.0)),
    Vector((0.22800704836845398, 0.9736595153808594, 0.0)),
    Vector((-0.9668625593185425, 0.25529745221138, 0.0)),
    Vector((-0.23333214223384857, -0.9723970890045166, 0.0)),
    Vector((0.9703369140625, -0.24175651371479034, 0.0)),
    Vector((0.38312801718711853, -0.9236952662467957, 0.0)),
    Vector((-0.881356418132782, -0.4724521338939667, 0.0)),
    Vector((0.4495607614517212, -0.8932497501373291, 0.0)),
    Vector((-0.8249465227127075, -0.5652108192443848, 0.0)),
    Vector((-0.6721270680427551, -0.740435779094696, 0.0)),
    Vector((-0.43356987833976746, -0.9011198878288269, 0.0)),
    Vector((-0.252834290266037, -0.9675096273422241, 0.0)),
    Vector((-0.1359875351190567, -0.9907105565071106, 0.0)),
    Vector((0.33476293087005615, -0.9423024654388428, 0.0)),
    Vector((-0.921581506729126, -0.38818472623825073, 0.0)),
    Vector((0.3834697902202606, -0.9235534071922302, 0.0)),
    Vector((0.9745244979858398, 0.2242812216281891, 0.0)),
    Vector((0.4335678517818451, -0.9011209607124329, 0.0)),
    Vector((0.5916162133216858, -0.8062196969985962, 0.0)),
    Vector((0.8580703735351562, -0.5135319232940674, 0.0)),
    Vector((-0.39827486872673035, -0.9172661900520325, 0.0)),
    Vector((0.943058967590332, -0.33262553811073303, 0.0)),
    Vector((0.4672276973724365, 0.8841371536254883, 0.0)),
    Vector((0.9399494528770447, -0.3413137197494507, 0.0)),
    Vector((0.9974864721298218, -0.07085702568292618, 0.0)),
    Vector((0.092048779129982, -0.9957544803619385, 0.0)),
    Vector((0.9994269609451294, -0.03384854272007942, 0.0)),
    Vector((-7.688333880651044e-07, 1.0, 0.0)),
    Vector((0.9827762842178345, 0.1847994178533554, 0.0)),
    Vector((0.9314835667610168, 0.3637833595275879, 0.0)),
    Vector((0.8623408079147339, 0.5063283443450928, 0.0)),
    Vector((0.5906959772109985, -0.8068942427635193, 0.0)),
    Vector((0.8051636815071106, 0.5930526852607727, 0.0)),
    Vector((0.7530662417411804, 0.6579446792602539, 0.0)),
    Vector((0.617594838142395, 0.7864964604377747, 0.0)),
    Vector((0.525052011013031, 0.8510701656341553, 0.0)),
    Vector((0.9997944831848145, 0.02027403563261032, 0.0)),
    Vector((3.6781405015062774e-06, -0.9999999403953552, 0.0)),
    Vector((0.9999417662620544, -0.010793542489409447, 0.0)),
    Vector((5.33354295839672e-06, -0.9999999403953552, 0.0)),
    Vector((0.999970018863678, 0.007746913004666567, 0.0)),
    Vector((-0.03293176367878914, 0.9994576573371887, 0.0)),
    Vector((0.9999364018440247, 0.011274782009422779, 0.0)),
    Vector((8.722763595869765e-06, -1.0, 0.0)),
    Vector((0.9999999403953552, 8.974959200713784e-06, 0.0)),
    Vector((-1.0306170224794187e-05, 0.9999999403953552, 0.0)),
    Vector((0.9999260306358337, 0.01216891035437584, 0.0)),
    Vector((0.956243634223938, 0.29257145524024963, 0.0)),
    Vector((0.9530967473983765, -0.3026657998561859, 0.0)),
    Vector((0.7080245614051819, -0.7061877846717834, 0.0)),
    Vector((0.2958528399467468, -0.9552335143089294, 0.0)),
    Vector((-0.25449156761169434, -0.967074990272522, 0.0)),
    Vector((-0.6469922661781311, -0.7624965906143188, 0.0)),
    Vector((-0.6180599331855774, 0.7861310243606567, 0.0)),
    Vector((-0.908999502658844, -0.41679704189300537, 0.0)),
    Vector((-0.2933608591556549, 0.9560017585754395, 0.0)),
    Vector((-0.984075129032135, 0.1777530461549759, 0.0)),
    Vector((-0.399539589881897, 0.9167159795761108, 0.0)),
    Vector((0.5438891649246216, 0.8391571044921875, 0.0)),
    Vector((-0.6856573820114136, 0.7279244065284729, 0.0)),
    Vector((0.77598637342453, 0.6307497024536133, 0.0)),
    Vector((-0.4462619721889496, 0.8949023485183716, 0.0)),
    Vector((-0.9713470935821533, 0.23766526579856873, 0.0)),
    Vector((0.23876070976257324, 0.9710784554481506, 0.0)),
    Vector((0.9713471531867981, -0.23766544461250305, 0.0)),
    Vector((-0.23876024782657623, -0.9710785746574402, 0.0)),
    Vector((-0.9712302088737488, 0.23814265429973602, 0.0)),
    Vector((0.23876020312309265, 0.9710785150527954, 0.0)),
    Vector((0.9712303280830383, -0.23814187943935394, 0.0)),
    Vector((-0.23875956237316132, -0.9710787534713745, 0.0)),
    Vector((-0.9707134962081909, 0.24023988842964172, 0.0)),
    Vector((0.23875927925109863, 0.9710787534713745, 0.0)),
    Vector((0.9707134962081909, -0.24024006724357605, 0.0)),
    Vector((-0.23875875771045685, -0.9710789322853088, 0.0)),
    Vector((-0.9712302684783936, 0.2381424754858017, 0.0)),
    Vector((0.23875856399536133, 0.9710789322853088, 0.0)),
    Vector((0.9712302088737488, -0.23814263939857483, 0.0)),
    Vector((-0.23875802755355835, -0.9710791110992432, 0.0))
]
holesInfo = [
    (145, 15),
    (164, 4),
    (172, 4),
    (180, 4),
    (188, 4)
]
firstVertIndex = 65
numPolygonVerts = 65

bpypolyskel.debugOutputs["skeleton"] = 1


faces = bpypolyskel.polygonize(verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, unitVectors)


# the number of vertices in a face
for face in faces:
    assert len(face) >= 3


# duplications of vertex indices
for face in faces:
    assert len(face) == len(set(face))


# edge crossing
assert not bpypolyskel.checkEdgeCrossing(bpypolyskel.debugOutputs["skeleton"])