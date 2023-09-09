import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((-0.04268435388803482, -0.31169456243515015, 0.0)),
    Vector((-0.36993107199668884, -0.26716676354408264, 0.0)),
    Vector((-0.5335545539855957, -1.4582853317260742, 0.0)),
    Vector((-0.26322025060653687, -1.5028131008148193, 0.0)),
    Vector((-0.7612050175666809, -5.087300777435303, 0.0)),
    Vector((-1.0244253873825073, -5.053905010223389, 0.0)),
    Vector((-1.1880491971969604, -6.256155014038086, 0.0)),
    Vector((-0.9177146553993225, -6.289551258087158, 0.0)),
    Vector((-1.4228143692016602, -9.92969799041748, 0.0)),
    Vector((-1.7073771953582764, -9.885170936584473, 0.0)),
    Vector((-1.8638871908187866, -11.031761169433594, 0.0)),
    Vector((-2.8171730041503906, -10.898177146911621, 0.0)),
    Vector((-2.866971731185913, -11.232135772705078, 0.0)),
    Vector((-3.1942191123962402, -11.209871292114258, 0.0)),
    Vector((-15.288389205932617, -96.21341705322266, 0.0)),
    Vector((-14.961136817932129, -96.28020477294922, 0.0)),
    Vector((-15.010937690734863, -96.62529754638672, 0.0)),
    Vector((-14.057636260986328, -96.75888061523438, 0.0)),
    Vector((-14.207036972045898, -97.87207794189453, 0.0)),
    Vector((-13.922470092773438, -97.9166030883789, 0.0)),
    Vector((-15.074992179870605, -106.29895782470703, 0.0)),
    Vector((-15.423588752746582, -106.25443267822266, 0.0)),
    Vector((-15.594332695007324, -107.47894287109375, 0.0)),
    Vector((-15.316879272460938, -107.51234436035156, 0.0)),
    Vector((-15.366679191589355, -107.84629821777344, 0.0)),
    Vector((-14.121694564819336, -108.0132827758789, 0.0)),
    Vector((-14.086122512817383, -107.75724792480469, 0.0)),
    Vector((-11.048358917236328, -108.16913604736328, 0.0)),
    Vector((-11.091044425964355, -108.4363021850586, 0.0)),
    Vector((-10.287139892578125, -108.54762268066406, 0.0)),
    Vector((-10.251567840576172, -108.29158782958984, 0.0)),
    Vector((-7.213803291320801, -108.71460723876953, 0.0)),
    Vector((-7.249374866485596, -108.99290466308594, 0.0)),
    Vector((-6.438355445861816, -109.10423278808594, 0.0)),
    Vector((-6.395669937133789, -108.825927734375, 0.0)),
    Vector((-3.372133255004883, -109.24894714355469, 0.0)),
    Vector((-3.4077045917510986, -109.5272445678711, 0.0)),
    Vector((-2.589570999145508, -109.63856506347656, 0.0)),
    Vector((-2.553999662399292, -109.38253021240234, 0.0)),
    Vector((0.4553091824054718, -109.7944107055664, 0.0)),
    Vector((0.41973817348480225, -110.07271575927734, 0.0)),
    Vector((1.2378718852996826, -110.1951675415039, 0.0)),
    Vector((1.2734428644180298, -109.93913269042969, 0.0)),
    Vector((4.325437545776367, -110.36214447021484, 0.0)),
    Vector((4.289866924285889, -110.61817932128906, 0.0)),
    Vector((5.463711261749268, -110.78515625, 0.0)),
    Vector((5.499281883239746, -110.49571990966797, 0.0)),
    Vector((5.983047962188721, -110.56251525878906, 0.0)),
    Vector((6.203586578369141, -108.98178100585938, 0.0)),
    Vector((5.783848762512207, -108.92611694335938, 0.0)),
    Vector((6.850969314575195, -101.18941497802734, 0.0)),
    Vector((7.3205060958862305, -101.2562026977539, 0.0)),
    Vector((7.526815891265869, -99.75338745117188, 0.0)),
    Vector((8.30937671661377, -99.86470794677734, 0.0)),
    Vector((8.35206127166748, -99.55301666259766, 0.0)),
    Vector((8.679314613342285, -99.59754180908203, 0.0)),
    Vector((11.994473457336426, -75.53025817871094, 0.0)),
    Vector((12.172327995300293, -75.55252075195312, 0.0)),
    Vector((12.279038429260254, -74.75102233886719, 0.0)),
    Vector((12.883742332458496, -74.82894897460938, 0.0)),
    Vector((12.848172187805176, -75.09611511230469, 0.0)),
    Vector((14.029123306274414, -75.26309204101562, 0.0)),
    Vector((14.064693450927734, -75.0181884765625, 0.0)),
    Vector((18.475473403930664, -75.61930084228516, 0.0)),
    Vector((18.432790756225586, -75.90872955322266, 0.0)),
    Vector((19.6208553314209, -76.0645751953125, 0.0)),
    Vector((19.649311065673828, -75.84193420410156, 0.0)),
    Vector((19.898305892944336, -75.87532806396484, 0.0)),
    Vector((19.940990447998047, -75.57476806640625, 0.0)),
    Vector((20.218442916870117, -75.61929321289062, 0.0)),
    Vector((20.382064819335938, -74.450439453125, 0.0)),
    Vector((20.140182495117188, -74.41703796386719, 0.0)),
    Vector((20.751983642578125, -69.98652648925781, 0.0)),
    Vector((21.043663024902344, -70.0199203491211, 0.0)),
    Vector((21.39936065673828, -67.43730163574219, 0.0)),
    Vector((21.12190818786621, -67.4039077758789, 0.0)),
    Vector((21.726593017578125, -62.98452377319336, 0.0)),
    Vector((21.996931076049805, -63.029048919677734, 0.0)),
    Vector((22.36685562133789, -60.35737991333008, 0.0)),
    Vector((22.06806182861328, -60.3128547668457, 0.0)),
    Vector((22.67985725402832, -55.8823356628418, 0.0)),
    Vector((23.007108688354492, -55.92686080932617, 0.0)),
    Vector((23.362804412841797, -53.35538101196289, 0.0)),
    Vector((23.128036499023438, -53.32198715209961, 0.0)),
    Vector((23.739831924438477, -48.924861907958984, 0.0)),
    Vector((24.024398803710938, -48.958255767822266, 0.0)),
    Vector((24.38009262084961, -46.342247009277344, 0.0)),
    Vector((24.145326614379883, -46.30885314941406, 0.0)),
    Vector((24.7500057220459, -41.88946533203125, 0.0)),
    Vector((25.041685104370117, -41.93399429321289, 0.0)),
    Vector((25.205303192138672, -40.73174285888672, 0.0)),
    Vector((24.927852630615234, -40.68721389770508, 0.0)),
    Vector((24.963422775268555, -40.40891647338867, 0.0)),
    Vector((24.707313537597656, -40.37552261352539, 0.0)),
    Vector((24.742883682250977, -40.130619049072266, 0.0)),
    Vector((23.533483505249023, -39.97477722167969, 0.0)),
    Vector((23.497913360595703, -40.230812072753906, 0.0)),
    Vector((19.094276428222656, -39.61857223510742, 0.0)),
    Vector((19.129846572875977, -39.35140609741211, 0.0)),
    Vector((17.920446395874023, -39.18442916870117, 0.0)),
    Vector((17.884876251220703, -39.44046401977539, 0.0)),
    Vector((17.3015193939209, -39.362545013427734, 0.0)),
    Vector((17.415342330932617, -38.54991149902344, 0.0)),
    Vector((17.15212059020996, -38.5053825378418, 0.0)),
    Vector((17.635868072509766, -34.98768615722656, 0.0)),
    Vector((17.906204223632812, -35.021080017089844, 0.0)),
    Vector((18.06982421875, -33.81883239746094, 0.0)),
    Vector((17.827943801879883, -33.78543472290039, 0.0)),
    Vector((18.311691284179688, -30.267738342285156, 0.0)),
    Vector((18.582027435302734, -30.312265396118164, 0.0)),
    Vector((18.752761840820312, -29.076618194580078, 0.0)),
    Vector((18.503767013549805, -29.043222427368164, 0.0)),
    Vector((18.98751449584961, -25.52552604675293, 0.0)),
    Vector((19.264963150024414, -25.55891990661621, 0.0)),
    Vector((19.4285831451416, -24.367801666259766, 0.0)),
    Vector((19.200931549072266, -24.33440589904785, 0.0)),
    Vector((19.691791534423828, -20.816707611083984, 0.0)),
    Vector((19.947898864746094, -20.8501033782959, 0.0)),
    Vector((20.11151885986328, -19.625587463378906, 0.0)),
    Vector((19.869638442993164, -19.592193603515625, 0.0)),
    Vector((20.353384017944336, -16.041099548339844, 0.0)),
    Vector((20.62371826171875, -16.074493408203125, 0.0)),
    Vector((20.837135314941406, -14.493756294250488, 0.0)),
    Vector((20.53122901916504, -14.449230194091797, 0.0)),
    Vector((20.573911666870117, -14.11527156829834, 0.0)),
    Vector((19.60639762878418, -13.981691360473633, 0.0)),
    Vector((19.755788803100586, -12.86849594116211, 0.0)),
    Vector((19.49256706237793, -12.835101127624512, 0.0)),
    Vector((19.99765396118164, -9.194952011108398, 0.0)),
    Vector((20.260873794555664, -9.228346824645996, 0.0)),
    Vector((20.42449378967285, -8.037227630615234, 0.0)),
    Vector((20.139930725097656, -8.003832817077637, 0.0)),
    Vector((20.630786895751953, -4.43047571182251, 0.0)),
    Vector((20.901121139526367, -4.463870525360107, 0.0)),
    Vector((21.071853637695312, -3.2504873275756836, 0.0)),
    Vector((20.758834838867188, -3.205960750579834, 0.0)),
    Vector((20.808631896972656, -2.860870122909546, 0.0)),
    Vector((19.592126846313477, -2.6938955783843994, 0.0)),
    Vector((19.55655860900879, -2.9721944332122803, 0.0)),
    Vector((16.959924697875977, -2.615980863571167, 0.0)),
    Vector((16.995492935180664, -2.348814010620117, 0.0)),
    Vector((16.17737579345703, -2.237497091293335, 0.0)),
    Vector((16.141807556152344, -2.5046639442443848, 0.0)),
    Vector((13.573629379272461, -2.1484487056732178, 0.0)),
    Vector((13.60208511352539, -1.892413854598999, 0.0)),
    Vector((12.819538116455078, -1.781096339225769, 0.0)),
    Vector((13.02584171295166, -0.31167855858802795, 0.0)),
    Vector((12.776849746704102, -0.27828332781791687, 0.0)),
    Vector((12.812419891357422, -0.011116459965705872, 0.0)),
    Vector((12.015645027160645, 0.10020116716623306, 0.0)),
    Vector((11.97296142578125, -0.178097665309906, 0.0)),
    Vector((9.234047889709473, 0.20038312673568726, 0.0)),
    Vector((9.276731491088867, 0.47868192195892334, 0.0)),
    Vector((8.45150089263916, 0.5900000333786011, 0.0)),
    Vector((8.40881633758545, 0.3005692958831787, 0.0)),
    Vector((8.145596504211426, 0.3450966775417328, 0.0)),
    Vector((7.946404933929443, -1.1131889820098877, 0.0)),
    Vector((7.213656902313232, -1.0130025148391724, 0.0)),
    Vector((7.178086757659912, -1.2690373659133911, 0.0)),
    Vector((4.588568687438965, -0.9128178358078003, 0.0)),
    Vector((4.624138832092285, -0.6456510424613953, 0.0)),
    Vector((3.7917935848236084, -0.5343322157859802, 0.0)),
    Vector((3.756223440170288, -0.790367066860199, 0.0)),
    Vector((1.1524776220321655, -0.43414589762687683, 0.0)),
    Vector((1.1880477666854858, -0.1669791042804718, 0.0)),
    Vector((0.0, 0.0, 8.527153968811035)),
    Vector((-0.04268435388803482, -0.31169456243515015, 8.527153968811035)),
    Vector((-0.36993107199668884, -0.26716676354408264, 8.527153968811035)),
    Vector((-0.5335545539855957, -1.4582853317260742, 8.527153968811035)),
    Vector((-0.26322025060653687, -1.5028131008148193, 8.527153968811035)),
    Vector((-0.7612050175666809, -5.087300777435303, 8.527153968811035)),
    Vector((-1.0244253873825073, -5.053905010223389, 8.527153968811035)),
    Vector((-1.1880491971969604, -6.256155014038086, 8.527153968811035)),
    Vector((-0.9177146553993225, -6.289551258087158, 8.527153968811035)),
    Vector((-1.4228143692016602, -9.92969799041748, 8.527153968811035)),
    Vector((-1.7073771953582764, -9.885170936584473, 8.527153968811035)),
    Vector((-1.8638871908187866, -11.031761169433594, 8.527153968811035)),
    Vector((-2.8171730041503906, -10.898177146911621, 8.527153968811035)),
    Vector((-2.866971731185913, -11.232135772705078, 8.527153968811035)),
    Vector((-3.1942191123962402, -11.209871292114258, 8.527153968811035)),
    Vector((-15.288389205932617, -96.21341705322266, 8.527153968811035)),
    Vector((-14.961136817932129, -96.28020477294922, 8.527153968811035)),
    Vector((-15.010937690734863, -96.62529754638672, 8.527153968811035)),
    Vector((-14.057636260986328, -96.75888061523438, 8.527153968811035)),
    Vector((-14.207036972045898, -97.87207794189453, 8.527153968811035)),
    Vector((-13.922470092773438, -97.9166030883789, 8.527153968811035)),
    Vector((-15.074992179870605, -106.29895782470703, 8.527153968811035)),
    Vector((-15.423588752746582, -106.25443267822266, 8.527153968811035)),
    Vector((-15.594332695007324, -107.47894287109375, 8.527153968811035)),
    Vector((-15.316879272460938, -107.51234436035156, 8.527153968811035)),
    Vector((-15.366679191589355, -107.84629821777344, 8.527153968811035)),
    Vector((-14.121694564819336, -108.0132827758789, 8.527153968811035)),
    Vector((-14.086122512817383, -107.75724792480469, 8.527153968811035)),
    Vector((-11.048358917236328, -108.16913604736328, 8.527153968811035)),
    Vector((-11.091044425964355, -108.4363021850586, 8.527153968811035)),
    Vector((-10.287139892578125, -108.54762268066406, 8.527153968811035)),
    Vector((-10.251567840576172, -108.29158782958984, 8.527153968811035)),
    Vector((-7.213803291320801, -108.71460723876953, 8.527153968811035)),
    Vector((-7.249374866485596, -108.99290466308594, 8.527153968811035)),
    Vector((-6.438355445861816, -109.10423278808594, 8.527153968811035)),
    Vector((-6.395669937133789, -108.825927734375, 8.527153968811035)),
    Vector((-3.372133255004883, -109.24894714355469, 8.527153968811035)),
    Vector((-3.4077045917510986, -109.5272445678711, 8.527153968811035)),
    Vector((-2.589570999145508, -109.63856506347656, 8.527153968811035)),
    Vector((-2.553999662399292, -109.38253021240234, 8.527153968811035)),
    Vector((0.4553091824054718, -109.7944107055664, 8.527153968811035)),
    Vector((0.41973817348480225, -110.07271575927734, 8.527153968811035)),
    Vector((1.2378718852996826, -110.1951675415039, 8.527153968811035)),
    Vector((1.2734428644180298, -109.93913269042969, 8.527153968811035)),
    Vector((4.325437545776367, -110.36214447021484, 8.527153968811035)),
    Vector((4.289866924285889, -110.61817932128906, 8.527153968811035)),
    Vector((5.463711261749268, -110.78515625, 8.527153968811035)),
    Vector((5.499281883239746, -110.49571990966797, 8.527153968811035)),
    Vector((5.983047962188721, -110.56251525878906, 8.527153968811035)),
    Vector((6.203586578369141, -108.98178100585938, 8.527153968811035)),
    Vector((5.783848762512207, -108.92611694335938, 8.527153968811035)),
    Vector((6.850969314575195, -101.18941497802734, 8.527153968811035)),
    Vector((7.3205060958862305, -101.2562026977539, 8.527153968811035)),
    Vector((7.526815891265869, -99.75338745117188, 8.527153968811035)),
    Vector((8.30937671661377, -99.86470794677734, 8.527153968811035)),
    Vector((8.35206127166748, -99.55301666259766, 8.527153968811035)),
    Vector((8.679314613342285, -99.59754180908203, 8.527153968811035)),
    Vector((11.994473457336426, -75.53025817871094, 8.527153968811035)),
    Vector((12.172327995300293, -75.55252075195312, 8.527153968811035)),
    Vector((12.279038429260254, -74.75102233886719, 8.527153968811035)),
    Vector((12.883742332458496, -74.82894897460938, 8.527153968811035)),
    Vector((12.848172187805176, -75.09611511230469, 8.527153968811035)),
    Vector((14.029123306274414, -75.26309204101562, 8.527153968811035)),
    Vector((14.064693450927734, -75.0181884765625, 8.527153968811035)),
    Vector((18.475473403930664, -75.61930084228516, 8.527153968811035)),
    Vector((18.432790756225586, -75.90872955322266, 8.527153968811035)),
    Vector((19.6208553314209, -76.0645751953125, 8.527153968811035)),
    Vector((19.649311065673828, -75.84193420410156, 8.527153968811035)),
    Vector((19.898305892944336, -75.87532806396484, 8.527153968811035)),
    Vector((19.940990447998047, -75.57476806640625, 8.527153968811035)),
    Vector((20.218442916870117, -75.61929321289062, 8.527153968811035)),
    Vector((20.382064819335938, -74.450439453125, 8.527153968811035)),
    Vector((20.140182495117188, -74.41703796386719, 8.527153968811035)),
    Vector((20.751983642578125, -69.98652648925781, 8.527153968811035)),
    Vector((21.043663024902344, -70.0199203491211, 8.527153968811035)),
    Vector((21.39936065673828, -67.43730163574219, 8.527153968811035)),
    Vector((21.12190818786621, -67.4039077758789, 8.527153968811035)),
    Vector((21.726593017578125, -62.98452377319336, 8.527153968811035)),
    Vector((21.996931076049805, -63.029048919677734, 8.527153968811035)),
    Vector((22.36685562133789, -60.35737991333008, 8.527153968811035)),
    Vector((22.06806182861328, -60.3128547668457, 8.527153968811035)),
    Vector((22.67985725402832, -55.8823356628418, 8.527153968811035)),
    Vector((23.007108688354492, -55.92686080932617, 8.527153968811035)),
    Vector((23.362804412841797, -53.35538101196289, 8.527153968811035)),
    Vector((23.128036499023438, -53.32198715209961, 8.527153968811035)),
    Vector((23.739831924438477, -48.924861907958984, 8.527153968811035)),
    Vector((24.024398803710938, -48.958255767822266, 8.527153968811035)),
    Vector((24.38009262084961, -46.342247009277344, 8.527153968811035)),
    Vector((24.145326614379883, -46.30885314941406, 8.527153968811035)),
    Vector((24.7500057220459, -41.88946533203125, 8.527153968811035)),
    Vector((25.041685104370117, -41.93399429321289, 8.527153968811035)),
    Vector((25.205303192138672, -40.73174285888672, 8.527153968811035)),
    Vector((24.927852630615234, -40.68721389770508, 8.527153968811035)),
    Vector((24.963422775268555, -40.40891647338867, 8.527153968811035)),
    Vector((24.707313537597656, -40.37552261352539, 8.527153968811035)),
    Vector((24.742883682250977, -40.130619049072266, 8.527153968811035)),
    Vector((23.533483505249023, -39.97477722167969, 8.527153968811035)),
    Vector((23.497913360595703, -40.230812072753906, 8.527153968811035)),
    Vector((19.094276428222656, -39.61857223510742, 8.527153968811035)),
    Vector((19.129846572875977, -39.35140609741211, 8.527153968811035)),
    Vector((17.920446395874023, -39.18442916870117, 8.527153968811035)),
    Vector((17.884876251220703, -39.44046401977539, 8.527153968811035)),
    Vector((17.3015193939209, -39.362545013427734, 8.527153968811035)),
    Vector((17.415342330932617, -38.54991149902344, 8.527153968811035)),
    Vector((17.15212059020996, -38.5053825378418, 8.527153968811035)),
    Vector((17.635868072509766, -34.98768615722656, 8.527153968811035)),
    Vector((17.906204223632812, -35.021080017089844, 8.527153968811035)),
    Vector((18.06982421875, -33.81883239746094, 8.527153968811035)),
    Vector((17.827943801879883, -33.78543472290039, 8.527153968811035)),
    Vector((18.311691284179688, -30.267738342285156, 8.527153968811035)),
    Vector((18.582027435302734, -30.312265396118164, 8.527153968811035)),
    Vector((18.752761840820312, -29.076618194580078, 8.527153968811035)),
    Vector((18.503767013549805, -29.043222427368164, 8.527153968811035)),
    Vector((18.98751449584961, -25.52552604675293, 8.527153968811035)),
    Vector((19.264963150024414, -25.55891990661621, 8.527153968811035)),
    Vector((19.4285831451416, -24.367801666259766, 8.527153968811035)),
    Vector((19.200931549072266, -24.33440589904785, 8.527153968811035)),
    Vector((19.691791534423828, -20.816707611083984, 8.527153968811035)),
    Vector((19.947898864746094, -20.8501033782959, 8.527153968811035)),
    Vector((20.11151885986328, -19.625587463378906, 8.527153968811035)),
    Vector((19.869638442993164, -19.592193603515625, 8.527153968811035)),
    Vector((20.353384017944336, -16.041099548339844, 8.527153968811035)),
    Vector((20.62371826171875, -16.074493408203125, 8.527153968811035)),
    Vector((20.837135314941406, -14.493756294250488, 8.527153968811035)),
    Vector((20.53122901916504, -14.449230194091797, 8.527153968811035)),
    Vector((20.573911666870117, -14.11527156829834, 8.527153968811035)),
    Vector((19.60639762878418, -13.981691360473633, 8.527153968811035)),
    Vector((19.755788803100586, -12.86849594116211, 8.527153968811035)),
    Vector((19.49256706237793, -12.835101127624512, 8.527153968811035)),
    Vector((19.99765396118164, -9.194952011108398, 8.527153968811035)),
    Vector((20.260873794555664, -9.228346824645996, 8.527153968811035)),
    Vector((20.42449378967285, -8.037227630615234, 8.527153968811035)),
    Vector((20.139930725097656, -8.003832817077637, 8.527153968811035)),
    Vector((20.630786895751953, -4.43047571182251, 8.527153968811035)),
    Vector((20.901121139526367, -4.463870525360107, 8.527153968811035)),
    Vector((21.071853637695312, -3.2504873275756836, 8.527153968811035)),
    Vector((20.758834838867188, -3.205960750579834, 8.527153968811035)),
    Vector((20.808631896972656, -2.860870122909546, 8.527153968811035)),
    Vector((19.592126846313477, -2.6938955783843994, 8.527153968811035)),
    Vector((19.55655860900879, -2.9721944332122803, 8.527153968811035)),
    Vector((16.959924697875977, -2.615980863571167, 8.527153968811035)),
    Vector((16.995492935180664, -2.348814010620117, 8.527153968811035)),
    Vector((16.17737579345703, -2.237497091293335, 8.527153968811035)),
    Vector((16.141807556152344, -2.5046639442443848, 8.527153968811035)),
    Vector((13.573629379272461, -2.1484487056732178, 8.527153968811035)),
    Vector((13.60208511352539, -1.892413854598999, 8.527153968811035)),
    Vector((12.819538116455078, -1.781096339225769, 8.527153968811035)),
    Vector((13.02584171295166, -0.31167855858802795, 8.527153968811035)),
    Vector((12.776849746704102, -0.27828332781791687, 8.527153968811035)),
    Vector((12.812419891357422, -0.011116459965705872, 8.527153968811035)),
    Vector((12.015645027160645, 0.10020116716623306, 8.527153968811035)),
    Vector((11.97296142578125, -0.178097665309906, 8.527153968811035)),
    Vector((9.234047889709473, 0.20038312673568726, 8.527153968811035)),
    Vector((9.276731491088867, 0.47868192195892334, 8.527153968811035)),
    Vector((8.45150089263916, 0.5900000333786011, 8.527153968811035)),
    Vector((8.40881633758545, 0.3005692958831787, 8.527153968811035)),
    Vector((8.145596504211426, 0.3450966775417328, 8.527153968811035)),
    Vector((7.946404933929443, -1.1131889820098877, 8.527153968811035)),
    Vector((7.213656902313232, -1.0130025148391724, 8.527153968811035)),
    Vector((7.178086757659912, -1.2690373659133911, 8.527153968811035)),
    Vector((4.588568687438965, -0.9128178358078003, 8.527153968811035)),
    Vector((4.624138832092285, -0.6456510424613953, 8.527153968811035)),
    Vector((3.7917935848236084, -0.5343322157859802, 8.527153968811035)),
    Vector((3.756223440170288, -0.790367066860199, 8.527153968811035)),
    Vector((1.1524776220321655, -0.43414589762687683, 8.527153968811035)),
    Vector((1.1880477666854858, -0.1669791042804718, 8.527153968811035))
]
unitVectors = [
    Vector((-0.13567660748958588, -0.990753173828125, 0.0)),
    Vector((-0.9908693432807922, 0.13482558727264404, 0.0)),
    Vector((-0.13609156012535095, -0.9906963109970093, 0.0)),
    Vector((0.9867046475410461, -0.1625237911939621, 0.0)),
    Vector((-0.13760612905025482, -0.9904870390892029, 0.0)),
    Vector((-0.9920474290847778, 0.12586481869220734, 0.0)),
    Vector((-0.1348547786474228, -0.9908653497695923, 0.0)),
    Vector((0.9924556016921997, -0.12260471284389496, 0.0)),
    Vector((-0.1374412328004837, -0.9905098676681519, 0.0)),
    Vector((-0.9879781007766724, 0.15459416806697845, 0.0)),
    Vector((-0.13524621725082397, -0.9908120632171631, 0.0)),
    Vector((-0.990324079990387, 0.13877420127391815, 0.0)),
    Vector((-0.14748574793338776, -0.9890641570091248, 0.0)),
    Vector((-0.997693657875061, 0.06787870824337006, 0.0)),
    Vector((-0.14085984230041504, -0.9900295734405518, 0.0)),
    Vector((0.9798031449317932, -0.19996437430381775, 0.0)),
    Vector((-0.1428319215774536, -0.9897469878196716, 0.0)),
    Vector((0.9903245568275452, -0.13877099752426147, 0.0)),
    Vector((-0.13301606476306915, -0.9911139011383057, 0.0)),
    Vector((0.9879794120788574, -0.1545855551958084, 0.0)),
    Vector((-0.13621234893798828, -0.9906796813011169, 0.0)),
    Vector((-0.9919413328170776, 0.12669755518436432, 0.0)),
    Vector((-0.1381024569272995, -0.9904179573059082, 0.0)),
    Vector((0.9928314089775085, -0.11952293664216995, 0.0)),
    Vector((-0.14749126136302948, -0.9890633821487427, 0.0)),
    Vector((0.99112468957901, -0.13293538987636566, 0.0)),
    Vector((0.13761259615421295, 0.9904860854148865, 0.0)),
    Vector((0.9909326434135437, -0.13435982167720795, 0.0)),
    Vector((-0.15777041018009186, -0.9874758720397949, 0.0)),
    Vector((0.9905480742454529, -0.13716591894626617, 0.0)),
    Vector((0.13761259615421295, 0.9904860854148865, 0.0)),
    Vector((0.9904430508613586, -0.13792268931865692, 0.0)),
    Vector((-0.12678705155849457, -0.9919299483299255, 0.0)),
    Vector((0.990709662437439, -0.135994091629982, 0.0)),
    Vector((0.15160386264324188, 0.9884413480758667, 0.0)),
    Vector((0.9903541207313538, -0.1385592669248581, 0.0)),
    Vector((-0.1267862170934677, -0.9919300675392151, 0.0)),
    Vector((0.9908695816993713, -0.13482405245304108, 0.0)),
    Vector((0.13760989904403687, 0.9904865622520447, 0.0)),
    Vector((0.9907630085945129, -0.13560454547405243, 0.0)),
    Vector((-0.1267816424369812, -0.9919306039810181, 0.0)),
    Vector((0.9889838695526123, -0.14802329242229462, 0.0)),
    Vector((0.13760854303836823, 0.9904866814613342, 0.0)),
    Vector((0.9905309081077576, -0.13728930056095123, 0.0)),
    Vector((-0.1376071721315384, -0.9904868602752686, 0.0)),
    Vector((0.990033745765686, -0.14083024859428406, 0.0)),
    Vector((0.12197848409414291, 0.9925327301025391, 0.0)),
    Vector((0.9906020164489746, -0.13677603006362915, 0.0)),
    Vector((0.1381782442331314, 0.990407407283783, 0.0)),
    Vector((-0.9913207292556763, 0.13146525621414185, 0.0)),
    Vector((0.13663603365421295, 0.9906212687492371, 0.0)),
    Vector((0.9900345206260681, -0.14082421362400055, 0.0)),
    Vector((0.13600657880306244, 0.9907079339027405, 0.0)),
    Vector((0.9900332689285278, -0.140833780169487, 0.0)),
    Vector((0.13567861914634705, 0.9907528758049011, 0.0)),
    Vector((0.9908707141876221, -0.1348150223493576, 0.0)),
    Vector((0.13645698130130768, 0.9906460046768188, 0.0)),
    Vector((0.992256760597229, -0.12420368194580078, 0.0)),
    Vector((0.13197413086891174, 0.9912531971931458, 0.0)),
    Vector((0.9917986392974854, -0.127810537815094, 0.0)),
    Vector((-0.13197413086891174, -0.991253137588501, 0.0)),
    Vector((0.990151584148407, -0.1399994194507599, 0.0)),
    Vector((0.14373332262039185, 0.9896165132522583, 0.0)),
    Vector((0.9908409714698792, -0.13503433763980865, 0.0)),
    Vector((-0.1458941400051117, -0.9893002510070801, 0.0)),
    Vector((0.9915059208869934, -0.13006184995174408, 0.0)),
    Vector((0.12677864730358124, 0.9919310212135315, 0.0)),
    Vector((0.9911261200904846, -0.13292455673217773, 0.0)),
    Vector((0.1406058967113495, 0.9900655746459961, 0.0)),
    Vector((0.987366795539856, -0.158451110124588, 0.0)),
    Vector((0.13863320648670197, 0.9903438091278076, 0.0)),
    Vector((-0.9905997514724731, 0.13679175078868866, 0.0)),
    Vector((0.13679012656211853, 0.990600049495697, 0.0)),
    Vector((0.9935099482536316, -0.11374520510435104, 0.0)),
    Vector((0.13643953204154968, 0.9906483292579651, 0.0)),
    Vector((-0.9928346872329712, 0.11949644237756729, 0.0)),
    Vector((0.13556252419948578, 0.9907687902450562, 0.0)),
    Vector((0.986706554889679, -0.16251227259635925, 0.0)),
    Vector((0.1371534764766693, 0.9905498027801514, 0.0)),
    Vector((-0.98907870054245, 0.14738884568214417, 0.0)),
    Vector((0.13678865134716034, 0.9906002879142761, 0.0)),
    Vector((0.9908707141876221, -0.13481579720973969, 0.0)),
    Vector((0.13701875507831573, 0.9905684590339661, 0.0)),
    Vector((-0.9900346398353577, 0.14082451164722443, 0.0)),
    Vector((0.13780781626701355, 0.9904589056968689, 0.0)),
    Vector((0.9931848049163818, -0.116550013422966, 0.0)),
    Vector((0.13472844660282135, 0.9908825159072876, 0.0)),
    Vector((-0.9900344610214233, 0.14082562923431396, 0.0)),
    Vector((0.13556115329265594, 0.9907690286636353, 0.0)),
    Vector((0.9885466694831848, -0.15091556310653687, 0.0)),
    Vector((0.13485001027584076, 0.9908660650253296, 0.0)),
    Vector((-0.9873645305633545, 0.1584654152393341, 0.0)),
    Vector((0.12678204476833344, 0.9919306039810181, 0.0)),
    Vector((-0.9916062355041504, 0.12929467856884003, 0.0)),
    Vector((0.14373332262039185, 0.9896165132522583, 0.0)),
    Vector((-0.9917997717857361, 0.12780210375785828, 0.0)),
    Vector((-0.13760536909103394, -0.9904870986938477, 0.0)),
    Vector((-0.990473210811615, 0.1377059817314148, 0.0)),
    Vector((0.13197413086891174, 0.991253137588501, 0.0)),
    Vector((-0.9906030893325806, 0.13676850497722626, 0.0)),
    Vector((-0.13760536909103394, -0.9904870986938477, 0.0)),
    Vector((-0.9911972284317017, 0.13239426910877228, 0.0)),
    Vector((0.13871267437934875, 0.9903326630592346, 0.0)),
    Vector((-0.9859908819198608, 0.16679909825325012, 0.0)),
    Vector((0.13623608648777008, 0.9906764030456543, 0.0)),
    Vector((0.9924567341804504, -0.12259536981582642, 0.0)),
    Vector((0.13485196232795715, 0.9908657073974609, 0.0)),
    Vector((-0.9906017780303955, 0.13677749037742615, 0.0)),
    Vector((0.13623608648777008, 0.9906764030456543, 0.0)),
    Vector((0.9867051839828491, -0.1625201553106308, 0.0)),
    Vector((0.13687364757061005, 0.9905885457992554, 0.0)),
    Vector((-0.9911251664161682, 0.13293202221393585, 0.0)),
    Vector((0.13623608648777008, 0.9906764030456543, 0.0)),
    Vector((0.9928344488143921, -0.11949805915355682, 0.0)),
    Vector((0.13608874380588531, 0.9906966686248779, 0.0)),
    Vector((-0.9894105792045593, 0.14514338970184326, 0.0)),
    Vector((0.13820111751556396, 0.9904042482376099, 0.0)),
    Vector((0.9916052222251892, -0.12930288910865784, 0.0)),
    Vector((0.13244304060935974, 0.9911906719207764, 0.0)),
    Vector((-0.9906039237976074, 0.13676215708255768, 0.0)),
    Vector((0.13497774302959442, 0.9908486604690552, 0.0)),
    Vector((0.9924565553665161, -0.12259621918201447, 0.0)),
    Vector((0.13379718363285065, 0.9910087585449219, 0.0)),
    Vector((-0.9895723462104797, 0.1440369039773941, 0.0)),
    Vector((0.12677693367004395, 0.9919312596321106, 0.0)),
    Vector((-0.9906031489372253, 0.13676801323890686, 0.0)),
    Vector((0.1330079436302185, 0.9911149740219116, 0.0)),
    Vector((-0.992047905921936, 0.12586063146591187, 0.0)),
    Vector((0.13743773102760315, 0.9905104041099548, 0.0)),
    Vector((0.9920478463172913, -0.1258615404367447, 0.0)),
    Vector((0.13608863949775696, 0.9906966686248779, 0.0)),
    Vector((-0.9931842684745789, 0.11655484139919281, 0.0)),
    Vector((0.13608761131763458, 0.9906968474388123, 0.0)),
    Vector((0.9924562573432922, -0.12259967625141144, 0.0)),
    Vector((0.1393352597951889, 0.9902453422546387, 0.0)),
    Vector((-0.9900336265563965, 0.14083118736743927, 0.0)),
    Vector((0.14282207190990448, 0.9897484183311462, 0.0)),
    Vector((-0.9907112121582031, 0.13598263263702393, 0.0)),
    Vector((-0.12677471339702606, -0.9919316172599792, 0.0)),
    Vector((-0.9907212257385254, 0.13590994477272034, 0.0)),
    Vector((0.13196682929992676, 0.9912541508674622, 0.0)),
    Vector((-0.9908697605133057, 0.13482245802879333, 0.0)),
    Vector((-0.13196682929992676, -0.9912541508674622, 0.0)),
    Vector((-0.990517258644104, 0.1373881846666336, 0.0)),
    Vector((0.11045996844768524, 0.993880569934845, 0.0)),
    Vector((-0.9900333881378174, 0.14083251357078552, 0.0)),
    Vector((0.13903456926345825, 0.9902875423431396, 0.0)),
    Vector((-0.991125226020813, 0.1329314261674881, 0.0)),
    Vector((0.1319737732410431, 0.9912531971931458, 0.0)),
    Vector((-0.990381121635437, 0.13836640119552612, 0.0)),
    Vector((-0.15160056948661804, -0.9884418845176697, 0.0)),
    Vector((-0.9905868768692017, 0.13688570261001587, 0.0)),
    Vector((0.15160056948661804, 0.9884418249130249, 0.0)),
    Vector((-0.9910241961479187, 0.13368256390094757, 0.0)),
    Vector((-0.14589951932430267, -0.9892993569374084, 0.0)),
    Vector((-0.9859917163848877, 0.1667945384979248, 0.0)),
    Vector((-0.1353362798690796, -0.9907997250556946, 0.0)),
    Vector((-0.9907819628715515, 0.13546667993068695, 0.0)),
    Vector((-0.13760536909103394, -0.9904870986938477, 0.0)),
    Vector((-0.9906706213951111, 0.13627871870994568, 0.0)),
    Vector((0.13197380304336548, 0.991253137588501, 0.0)),
    Vector((-0.9911748170852661, 0.1325608789920807, 0.0)),
    Vector((-0.13760536909103394, -0.9904870986938477, 0.0)),
    Vector((-0.990770697593689, 0.13554836809635162, 0.0)),
    Vector((0.13197380304336548, 0.991253137588501, 0.0)),
    Vector((-0.9902668595314026, 0.13918116688728333, 0.0))
]
holesInfo = None
firstVertIndex = 166
numPolygonVerts = 166
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
