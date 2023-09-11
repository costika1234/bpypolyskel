import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((0.0, 0.0, 0.0)),
    Vector((1.6399973630905151, -2.6827995777130127, 0.0)),
    Vector((0.3092980980873108, -8.549336433410645, 0.0)),
    Vector((-2.150700569152832, -8.026134490966797, 0.0)),
    Vector((-2.647017002105713, -10.130072593688965, 0.0)),
    Vector((-0.1366666555404663, -10.731199264526367, 0.0)),
    Vector((-0.4963158667087555, -12.23401165008545, 0.0)),
    Vector((0.007192985154688358, -13.146831512451172, 0.0)),
    Vector((0.49631601572036743, -13.781352996826172, 0.0)),
    Vector((0.028771955519914627, -16.208118438720703, 0.0)),
    Vector((-1.4889490604400635, -17.209993362426758, 0.0)),
    Vector((-1.7191250324249268, -18.27865982055664, 0.0)),
    Vector((-4.330180644989014, -17.666400909423828, 0.0)),
    Vector((-4.804919719696045, -19.636756896972656, 0.0)),
    Vector((-2.2514073848724365, -20.271278381347656, 0.0)),
    Vector((-3.7691335678100586, -26.51630210876465, 0.0)),
    Vector((-5.912649631500244, -27.75194549560547, 0.0)),
    Vector((-5.056683540344238, -29.366079330444336, 0.0)),
    Vector((-2.539130687713623, -28.019115447998047, 0.0)),
    Vector((3.690012216567993, -29.27702522277832, 0.0)),
    Vector((3.229661464691162, -31.859638214111328, 0.0)),
    Vector((4.955984115600586, -32.171329498291016, 0.0)),
    Vector((5.473877906799316, -29.42173957824707, 0.0)),
    Vector((11.947587013244629, -30.73529815673828, 0.0)),
    Vector((11.559168815612793, -32.67225646972656, 0.0)),
    Vector((12.83952522277832, -32.92829132080078, 0.0)),
    Vector((12.7316312789917, -33.529415130615234, 0.0)),
    Vector((11.904435157775879, -33.37356948852539, 0.0)),
    Vector((11.760576248168945, -34.04148864746094, 0.0)),
    Vector((12.537422180175781, -34.609214782714844, 0.0)),
    Vector((15.882184982299805, -38.31614685058594, 0.0)),
    Vector((20.126054763793945, -35.343902587890625, 0.0)),
    Vector((20.902902603149414, -35.8448371887207, 0.0)),
    Vector((21.23377799987793, -35.25484085083008, 0.0)),
    Vector((20.485702514648438, -34.620323181152344, 0.0)),
    Vector((20.492895126342773, -34.44221115112305, 0.0)),
    Vector((21.917112350463867, -34.72050476074219, 0.0)),
    Vector((22.38465118408203, -32.928260803222656, 0.0)),
    Vector((28.419593811035156, -34.19727325439453, 0.0)),
    Vector((28.03118133544922, -36.0340461730957, 0.0)),
    Vector((29.160486221313477, -36.22328186035156, 0.0)),
    Vector((29.62082862854004, -34.45330047607422, 0.0)),
    Vector((36.59086990356445, -35.96720504760742, 0.0)),
    Vector((37.61228942871094, -37.44774627685547, 0.0)),
    Vector((38.532989501953125, -36.779823303222656, 0.0)),
    Vector((38.72000503540039, -36.37907028198242, 0.0)),
    Vector((39.86369705200195, -36.6462287902832, 0.0)),
    Vector((39.25230026245117, -38.08225631713867, 0.0)),
    Vector((40.14423751831055, -38.27149200439453, 0.0)),
    Vector((40.511070251464844, -36.79093933105469, 0.0)),
    Vector((42.81283950805664, -37.414310455322266, 0.0)),
    Vector((43.15811538696289, -38.83919906616211, 0.0)),
    Vector((43.949344635009766, -38.72787094116211, 0.0)),
    Vector((43.78389358520508, -37.25845718383789, 0.0)),
    Vector((45.89863204956055, -35.98939514160156, 0.0)),
    Vector((47.15022277832031, -36.79088592529297, 0.0)),
    Vector((47.69688415527344, -36.05617141723633, 0.0)),
    Vector((46.337398529052734, -35.26581573486328, 0.0)),
    Vector((46.912818908691406, -32.95036697387695, 0.0)),
    Vector((48.48808670043945, -32.605262756347656, 0.0)),
    Vector((48.38739013671875, -33.07280349731445, 0.0)),
    Vector((49.43756866455078, -33.273170471191406, 0.0)),
    Vector((49.84755325317383, -31.358470916748047, 0.0)),
    Vector((52.22124481201172, -31.102413177490234, 0.0)),
    Vector((52.99810791015625, -32.972572326660156, 0.0)),
    Vector((53.8972282409668, -32.50502395629883, 0.0)),
    Vector((53.82529830932617, -32.304649353027344, 0.0)),
    Vector((56.788795471191406, -30.267471313476562, 0.0)),
    Vector((57.01177978515625, -30.401052474975586, 0.0)),
    Vector((57.421775817871094, -29.666339874267578, 0.0)),
    Vector((57.105281829833984, -29.510494232177734, 0.0)),
    Vector((57.17001724243164, -29.376911163330078, 0.0)),
    Vector((55.558773040771484, -28.319393157958984, 0.0)),
    Vector((56.062259674072266, -26.015073776245117, 0.0)),
    Vector((58.01875305175781, -25.937129974365234, 0.0)),
    Vector((57.96120071411133, -24.924123764038086, 0.0)),
    Vector((56.04066848754883, -25.035463333129883, 0.0)),
    Vector((55.220645904541016, -22.80908203125, 0.0)),
    Vector((56.66642379760742, -21.517759323120117, 0.0)),
    Vector((56.026241302490234, -20.805322647094727, 0.0)),
    Vector((54.494144439697266, -22.05211639404297, 0.0)),
    Vector((52.501670837402344, -20.627246856689453, 0.0)),
    Vector((53.01954650878906, -18.71254539489746, 0.0)),
    Vector((52.041297912597656, -18.467653274536133, 0.0)),
    Vector((51.54499816894531, -20.382352828979492, 0.0)),
    Vector((48.7325325012207, -19.803518295288086, 0.0)),
    Vector((49.29357147216797, -17.955608367919922, 0.0)),
    Vector((49.545326232910156, -18.044662475585938, 0.0)),
    Vector((50.99107360839844, -13.480549812316895, 0.0)),
    Vector((52.753360748291016, -14.02599811553955, 0.0)),
    Vector((53.05545425415039, -13.001855850219727, 0.0)),
    Vector((51.40825653076172, -12.456406593322754, 0.0)),
    Vector((51.969295501708984, -10.886795997619629, 0.0)),
    Vector((50.88315200805664, -10.575112342834473, 0.0)),
    Vector((50.228607177734375, -12.456417083740234, 0.0)),
    Vector((48.51667404174805, -11.922099113464355, 0.0)),
    Vector((48.588600158691406, -11.632668495178223, 0.0)),
    Vector((47.970001220703125, -11.487957954406738, 0.0)),
    Vector((48.2720947265625, -10.28570556640625, 0.0)),
    Vector((45.919986724853516, -9.717996597290039, 0.0)),
    Vector((45.65385818481445, -10.875720977783203, 0.0)),
    Vector((44.99209976196289, -10.753274917602539, 0.0)),
    Vector((44.94175338745117, -11.076102256774902, 0.0)),
    Vector((43.057186126708984, -10.630839347839355, 0.0)),
    Vector((43.539100646972656, -8.60482120513916, 0.0)),
    Vector((42.09331130981445, -8.237478256225586, 0.0)),
    Vector((41.517887115478516, -10.274629592895508, 0.0)),
    Vector((35.382266998291016, -8.92770767211914, 0.0)),
    Vector((35.90734100341797, -6.868293285369873, 0.0)),
    Vector((34.432777404785156, -6.53434419631958, 0.0)),
    Vector((33.900508880615234, -8.660550117492676, 0.0)),
    Vector((28.347522735595703, -7.191164493560791, 0.0)),
    Vector((28.800668716430664, -5.2208075523376465, 0.0)),
    Vector((28.27558135986328, -5.020434856414795, 0.0)),
    Vector((28.491365432739258, -4.074218273162842, 0.0)),
    Vector((29.225046157836914, -3.662332057952881, 0.0)),
    Vector((28.750307083129883, -2.949889898300171, 0.0)),
    Vector((27.980661392211914, -3.43969988822937, 0.0)),
    Vector((24.218732833862305, -2.5714259147644043, 0.0)),
    Vector((24.31943130493164, -2.0482239723205566, 0.0)),
    Vector((23.355573654174805, -1.8478530645370483, 0.0)),
    Vector((23.240488052368164, -2.3487913608551025, 0.0)),
    Vector((19.57206916809082, -1.5250415802001953, 0.0)),
    Vector((19.176454544067383, -0.7569384574890137, 0.0)),
    Vector((18.284528732299805, -1.3580667972564697, 0.0)),
    Vector((18.72330093383789, -2.092773914337158, 0.0)),
    Vector((18.50751495361328, -3.0278584957122803, 0.0)),
    Vector((18.12628746032715, -2.9499361515045166, 0.0)),
    Vector((17.57243537902832, -4.931424617767334, 0.0)),
    Vector((11.616650581359863, -3.573343276977539, 0.0)),
    Vector((12.163310050964355, -0.7458269000053406, 0.0)),
    Vector((9.897523880004883, -0.3116855025291443, 0.0)),
    Vector((9.422792434692383, -2.8275067806243896, 0.0)),
    Vector((3.423853635787964, -1.5918676853179932, 0.0)),
    Vector((1.7335047721862793, 1.146591067314148, 0.0)),
    Vector((0.0, 0.0, 30.0)),
    Vector((1.6399973630905151, -2.6827995777130127, 30.0)),
    Vector((0.3092980980873108, -8.549336433410645, 30.0)),
    Vector((-2.150700569152832, -8.026134490966797, 30.0)),
    Vector((-2.647017002105713, -10.130072593688965, 30.0)),
    Vector((-0.1366666555404663, -10.731199264526367, 30.0)),
    Vector((-0.4963158667087555, -12.23401165008545, 30.0)),
    Vector((0.007192985154688358, -13.146831512451172, 30.0)),
    Vector((0.49631601572036743, -13.781352996826172, 30.0)),
    Vector((0.028771955519914627, -16.208118438720703, 30.0)),
    Vector((-1.4889490604400635, -17.209993362426758, 30.0)),
    Vector((-1.7191250324249268, -18.27865982055664, 30.0)),
    Vector((-4.330180644989014, -17.666400909423828, 30.0)),
    Vector((-4.804919719696045, -19.636756896972656, 30.0)),
    Vector((-2.2514073848724365, -20.271278381347656, 30.0)),
    Vector((-3.7691335678100586, -26.51630210876465, 30.0)),
    Vector((-5.912649631500244, -27.75194549560547, 30.0)),
    Vector((-5.056683540344238, -29.366079330444336, 30.0)),
    Vector((-2.539130687713623, -28.019115447998047, 30.0)),
    Vector((3.690012216567993, -29.27702522277832, 30.0)),
    Vector((3.229661464691162, -31.859638214111328, 30.0)),
    Vector((4.955984115600586, -32.171329498291016, 30.0)),
    Vector((5.473877906799316, -29.42173957824707, 30.0)),
    Vector((11.947587013244629, -30.73529815673828, 30.0)),
    Vector((11.559168815612793, -32.67225646972656, 30.0)),
    Vector((12.83952522277832, -32.92829132080078, 30.0)),
    Vector((12.7316312789917, -33.529415130615234, 30.0)),
    Vector((11.904435157775879, -33.37356948852539, 30.0)),
    Vector((11.760576248168945, -34.04148864746094, 30.0)),
    Vector((12.537422180175781, -34.609214782714844, 30.0)),
    Vector((15.882184982299805, -38.31614685058594, 30.0)),
    Vector((20.126054763793945, -35.343902587890625, 30.0)),
    Vector((20.902902603149414, -35.8448371887207, 30.0)),
    Vector((21.23377799987793, -35.25484085083008, 30.0)),
    Vector((20.485702514648438, -34.620323181152344, 30.0)),
    Vector((20.492895126342773, -34.44221115112305, 30.0)),
    Vector((21.917112350463867, -34.72050476074219, 30.0)),
    Vector((22.38465118408203, -32.928260803222656, 30.0)),
    Vector((28.419593811035156, -34.19727325439453, 30.0)),
    Vector((28.03118133544922, -36.0340461730957, 30.0)),
    Vector((29.160486221313477, -36.22328186035156, 30.0)),
    Vector((29.62082862854004, -34.45330047607422, 30.0)),
    Vector((36.59086990356445, -35.96720504760742, 30.0)),
    Vector((37.61228942871094, -37.44774627685547, 30.0)),
    Vector((38.532989501953125, -36.779823303222656, 30.0)),
    Vector((38.72000503540039, -36.37907028198242, 30.0)),
    Vector((39.86369705200195, -36.6462287902832, 30.0)),
    Vector((39.25230026245117, -38.08225631713867, 30.0)),
    Vector((40.14423751831055, -38.27149200439453, 30.0)),
    Vector((40.511070251464844, -36.79093933105469, 30.0)),
    Vector((42.81283950805664, -37.414310455322266, 30.0)),
    Vector((43.15811538696289, -38.83919906616211, 30.0)),
    Vector((43.949344635009766, -38.72787094116211, 30.0)),
    Vector((43.78389358520508, -37.25845718383789, 30.0)),
    Vector((45.89863204956055, -35.98939514160156, 30.0)),
    Vector((47.15022277832031, -36.79088592529297, 30.0)),
    Vector((47.69688415527344, -36.05617141723633, 30.0)),
    Vector((46.337398529052734, -35.26581573486328, 30.0)),
    Vector((46.912818908691406, -32.95036697387695, 30.0)),
    Vector((48.48808670043945, -32.605262756347656, 30.0)),
    Vector((48.38739013671875, -33.07280349731445, 30.0)),
    Vector((49.43756866455078, -33.273170471191406, 30.0)),
    Vector((49.84755325317383, -31.358470916748047, 30.0)),
    Vector((52.22124481201172, -31.102413177490234, 30.0)),
    Vector((52.99810791015625, -32.972572326660156, 30.0)),
    Vector((53.8972282409668, -32.50502395629883, 30.0)),
    Vector((53.82529830932617, -32.304649353027344, 30.0)),
    Vector((56.788795471191406, -30.267471313476562, 30.0)),
    Vector((57.01177978515625, -30.401052474975586, 30.0)),
    Vector((57.421775817871094, -29.666339874267578, 30.0)),
    Vector((57.105281829833984, -29.510494232177734, 30.0)),
    Vector((57.17001724243164, -29.376911163330078, 30.0)),
    Vector((55.558773040771484, -28.319393157958984, 30.0)),
    Vector((56.062259674072266, -26.015073776245117, 30.0)),
    Vector((58.01875305175781, -25.937129974365234, 30.0)),
    Vector((57.96120071411133, -24.924123764038086, 30.0)),
    Vector((56.04066848754883, -25.035463333129883, 30.0)),
    Vector((55.220645904541016, -22.80908203125, 30.0)),
    Vector((56.66642379760742, -21.517759323120117, 30.0)),
    Vector((56.026241302490234, -20.805322647094727, 30.0)),
    Vector((54.494144439697266, -22.05211639404297, 30.0)),
    Vector((52.501670837402344, -20.627246856689453, 30.0)),
    Vector((53.01954650878906, -18.71254539489746, 30.0)),
    Vector((52.041297912597656, -18.467653274536133, 30.0)),
    Vector((51.54499816894531, -20.382352828979492, 30.0)),
    Vector((48.7325325012207, -19.803518295288086, 30.0)),
    Vector((49.29357147216797, -17.955608367919922, 30.0)),
    Vector((49.545326232910156, -18.044662475585938, 30.0)),
    Vector((50.99107360839844, -13.480549812316895, 30.0)),
    Vector((52.753360748291016, -14.02599811553955, 30.0)),
    Vector((53.05545425415039, -13.001855850219727, 30.0)),
    Vector((51.40825653076172, -12.456406593322754, 30.0)),
    Vector((51.969295501708984, -10.886795997619629, 30.0)),
    Vector((50.88315200805664, -10.575112342834473, 30.0)),
    Vector((50.228607177734375, -12.456417083740234, 30.0)),
    Vector((48.51667404174805, -11.922099113464355, 30.0)),
    Vector((48.588600158691406, -11.632668495178223, 30.0)),
    Vector((47.970001220703125, -11.487957954406738, 30.0)),
    Vector((48.2720947265625, -10.28570556640625, 30.0)),
    Vector((45.919986724853516, -9.717996597290039, 30.0)),
    Vector((45.65385818481445, -10.875720977783203, 30.0)),
    Vector((44.99209976196289, -10.753274917602539, 30.0)),
    Vector((44.94175338745117, -11.076102256774902, 30.0)),
    Vector((43.057186126708984, -10.630839347839355, 30.0)),
    Vector((43.539100646972656, -8.60482120513916, 30.0)),
    Vector((42.09331130981445, -8.237478256225586, 30.0)),
    Vector((41.517887115478516, -10.274629592895508, 30.0)),
    Vector((35.382266998291016, -8.92770767211914, 30.0)),
    Vector((35.90734100341797, -6.868293285369873, 30.0)),
    Vector((34.432777404785156, -6.53434419631958, 30.0)),
    Vector((33.900508880615234, -8.660550117492676, 30.0)),
    Vector((28.347522735595703, -7.191164493560791, 30.0)),
    Vector((28.800668716430664, -5.2208075523376465, 30.0)),
    Vector((28.27558135986328, -5.020434856414795, 30.0)),
    Vector((28.491365432739258, -4.074218273162842, 30.0)),
    Vector((29.225046157836914, -3.662332057952881, 30.0)),
    Vector((28.750307083129883, -2.949889898300171, 30.0)),
    Vector((27.980661392211914, -3.43969988822937, 30.0)),
    Vector((24.218732833862305, -2.5714259147644043, 30.0)),
    Vector((24.31943130493164, -2.0482239723205566, 30.0)),
    Vector((23.355573654174805, -1.8478530645370483, 30.0)),
    Vector((23.240488052368164, -2.3487913608551025, 30.0)),
    Vector((19.57206916809082, -1.5250415802001953, 30.0)),
    Vector((19.176454544067383, -0.7569384574890137, 30.0)),
    Vector((18.284528732299805, -1.3580667972564697, 30.0)),
    Vector((18.72330093383789, -2.092773914337158, 30.0)),
    Vector((18.50751495361328, -3.0278584957122803, 30.0)),
    Vector((18.12628746032715, -2.9499361515045166, 30.0)),
    Vector((17.57243537902832, -4.931424617767334, 30.0)),
    Vector((11.616650581359863, -3.573343276977539, 30.0)),
    Vector((12.163310050964355, -0.7458269000053406, 30.0)),
    Vector((9.897523880004883, -0.3116855025291443, 30.0)),
    Vector((9.422792434692383, -2.8275067806243896, 30.0)),
    Vector((3.423853635787964, -1.5918676853179932, 30.0)),
    Vector((1.7335047721862793, 1.146591067314148, 30.0))
]
unitVectors = [
    Vector((0.5215678215026855, -0.8532098531723022, 0.0)),
    Vector((-0.22120937705039978, -0.9752264022827148, 0.0)),
    Vector((-0.9781222939491272, 0.2080307900905609, 0.0)),
    Vector((-0.22959692776203156, -0.9732857346534729, 0.0)),
    Vector((0.9725064635276794, -0.23287570476531982, 0.0)),
    Vector((-0.2327452450990677, -0.9725377559661865, 0.0)),
    Vector((0.48299211263656616, -0.8756247162818909, 0.0)),
    Vector((0.6105177998542786, -0.7920024991035461, 0.0)),
    Vector((-0.189182311296463, -0.9819418787956238, 0.0)),
    Vector((-0.8345641493797302, -0.5509107708930969, 0.0)),
    Vector((-0.21055752038955688, -0.9775814414024353, 0.0)),
    Vector((-0.9735919833183289, 0.22829478979110718, 0.0)),
    Vector((-0.2342376410961151, -0.9721793532371521, 0.0)),
    Vector((0.9704863429069519, -0.24115584790706635, 0.0)),
    Vector((-0.2361556589603424, -0.9717152714729309, 0.0)),
    Vector((-0.8663605451583862, -0.4994190037250519, 0.0)),
    Vector((0.4684966504573822, -0.8834652304649353, 0.0)),
    Vector((0.8817313313484192, 0.471751868724823, 0.0)),
    Vector((0.9802134037017822, -0.1979437619447708, 0.0)),
    Vector((-0.17548397183418274, -0.9844823479652405, 0.0)),
    Vector((0.9840884804725647, -0.17767930030822754, 0.0)),
    Vector((0.18509836494922638, 0.9827200174331665, 0.0)),
    Vector((0.9800290465354919, -0.19885440170764923, 0.0)),
    Vector((-0.19661574065685272, -0.9804806113243103, 0.0)),
    Vector((0.980586051940918, -0.19608931243419647, 0.0)),
    Vector((-0.17666395008563995, -0.9842712879180908, 0.0)),
    Vector((-0.9827112555503845, 0.1851450502872467, 0.0)),
    Vector((-0.21055525541305542, -0.9775820374488831, 0.0)),
    Vector((0.8073757886886597, -0.5900376439094543, 0.0)),
    Vector((0.6699074506759644, -0.7424446940422058, 0.0)),
    Vector((0.8190926909446716, 0.573661208152771, 0.0)),
    Vector((0.8404235243797302, -0.541930079460144, 0.0)),
    Vector((0.4891405999660492, 0.8722049593925476, 0.0)),
    Vector((-0.7626162171363831, 0.646851122379303, 0.0)),
    Vector((0.040349628776311874, 0.9991856813430786, 0.0)),
    Vector((0.9814390540122986, -0.19177427887916565, 0.0)),
    Vector((0.2524203360080719, 0.9676176905632019, 0.0)),
    Vector((0.978598952293396, -0.20577730238437653, 0.0)),
    Vector((-0.20688943564891815, -0.9783642888069153, 0.0)),
    Vector((0.9862493872642517, -0.16526411473751068, 0.0)),
    Vector((0.2517092823982239, 0.9678029417991638, 0.0)),
    Vector((0.9772148728370667, -0.2122526913881302, 0.0)),
    Vector((0.5678672790527344, -0.8231201171875, 0.0)),
    Vector((0.8094368577003479, 0.5872069597244263, 0.0)),
    Vector((0.42288070917129517, 0.9061852693557739, 0.0)),
    Vector((0.9737852215766907, -0.22746945917606354, 0.0)),
    Vector((-0.3917294144630432, -0.9200804233551025, 0.0)),
    Vector((0.9782259464263916, -0.20754291117191315, 0.0)),
    Vector((0.24049551784992218, 0.9706502556800842, 0.0)),
    Vector((0.9652290344238281, -0.2614058256149292, 0.0)),
    Vector((0.23550231754779816, -0.9718738198280334, 0.0)),
    Vector((0.9902459383010864, 0.13933032751083374, 0.0)),
    Vector((-0.11188960075378418, 0.9937206506729126, 0.0)),
    Vector((0.8574537634849548, 0.5145610570907593, 0.0)),
    Vector((0.8421269655227661, -0.5392792820930481, 0.0)),
    Vector((0.59693843126297, 0.8022870421409607, 0.0)),
    Vector((-0.8645190596580505, 0.5026000738143921, 0.0)),
    Vector((0.24117766320705414, 0.9704809188842773, 0.0)),
    Vector((0.9768333435058594, 0.21400126814842224, 0.0)),
    Vector((-0.2105470448732376, -0.9775837063789368, 0.0)),
    Vector((0.9822812080383301, -0.18741263449192047, 0.0)),
    Vector((0.20937861502170563, 0.9778346419334412, 0.0)),
    Vector((0.9942318797111511, 0.10725098848342896, 0.0)),
    Vector((0.383618026971817, -0.9234918355941772, 0.0)),
    Vector((0.8872144222259521, 0.4613572359085083, 0.0)),
    Vector((-0.33786720037460327, 0.9411938190460205, 0.0)),
    Vector((0.8240713477134705, 0.566486120223999, 0.0)),
    Vector((0.8578481078147888, -0.5139032006263733, 0.0)),
    Vector((0.4872971773147583, 0.8732361793518066, 0.0)),
    Vector((-0.8971334099769592, 0.4417598247528076, 0.0)),
    Vector((0.43609821796417236, 0.8998990654945374, 0.0)),
    Vector((-0.8360148668289185, 0.5487068891525269, 0.0)),
    Vector((0.21346087753772736, 0.976951539516449, 0.0)),
    Vector((0.9992073774337769, 0.03980694338679314, 0.0)),
    Vector((-0.056721944361925125, 0.9983900785446167, 0.0)),
    Vector((-0.9983236789703369, -0.057876106351614, 0.0)),
    Vector((-0.34562253952026367, 0.9383736252784729, 0.0)),
    Vector((0.7458224892616272, 0.666144847869873, 0.0)),
    Vector((-0.6683816909790039, 0.7438185811042786, 0.0)),
    Vector((-0.7756268978118896, -0.6311916708946228, 0.0)),
    Vector((-0.8134101033210754, 0.5816906094551086, 0.0)),
    Vector((0.2610917091369629, 0.9653140306472778, 0.0)),
    Vector((-0.9700655341148376, 0.24284358322620392, 0.0)),
    Vector((-0.25091299414634705, -0.9680097103118896, 0.0)),
    Vector((-0.9794709086418152, 0.20158524811267853, 0.0)),
    Vector((0.2905130684375763, 0.9568710327148438, 0.0)),
    Vector((0.9427556991577148, -0.33348432183265686, 0.0)),
    Vector((0.30197617411613464, 0.9533154368400574, 0.0)),
    Vector((0.9552892446517944, -0.29567310214042664, 0.0)),
    Vector((0.28292062878608704, 0.9591432809829712, 0.0)),
    Vector((-0.9493066668510437, 0.31435123085975647, 0.0)),
    Vector((0.33658313751220703, 0.9416537284851074, 0.0)),
    Vector((-0.9612060785293579, 0.2758311629295349, 0.0)),
    Vector((-0.32860031723976135, -0.9444690942764282, 0.0)),
    Vector((-0.9545848965644836, 0.29793912172317505, 0.0)),
    Vector((0.24117355048656464, 0.9704821109771729, 0.0)),
    Vector((-0.9737119078636169, 0.22778308391571045, 0.0)),
    Vector((0.2436974048614502, 0.9698513150215149, 0.0)),
    Vector((-0.9720860719680786, 0.23462443053722382, 0.0)),
    Vector((-0.2240293025970459, -0.9745823740959167, 0.0)),
    Vector((-0.9833090901374817, 0.18194301426410675, 0.0)),
    Vector((-0.15409184992313385, -0.9880565404891968, 0.0)),
    Vector((-0.9732054471969604, 0.22993728518486023, 0.0)),
    Vector((0.23140659928321838, 0.9728571176528931, 0.0)),
    Vector((-0.9692054390907288, 0.24625356495380402, 0.0)),
    Vector((-0.2718290388584137, -0.962345540523529, 0.0)),
    Vector((-0.9767417311668396, 0.21441921591758728, 0.0)),
    Vector((0.24705901741981506, 0.9690003395080566, 0.0)),
    Vector((-0.9753010869026184, 0.22087952494621277, 0.0)),
    Vector((-0.24284352362155914, -0.9700655341148376, 0.0)),
    Vector((-0.9667277336120605, 0.2558075487613678, 0.0)),
    Vector((0.22413073480129242, 0.9745591282844543, 0.0)),
    Vector((-0.9342866539955139, 0.35652264952659607, 0.0)),
    Vector((0.22234103083610535, 0.9749689698219299, 0.0)),
    Vector((0.8719862103462219, 0.4895305037498474, 0.0)),
    Vector((-0.5545203685760498, 0.8321701288223267, 0.0)),
    Vector((-0.843643844127655, -0.5369031429290771, 0.0)),
    Vector((-0.9743834733963013, 0.22489310801029205, 0.0)),
    Vector((0.18899710476398468, 0.9819777011871338, 0.0)),
    Vector((-0.9790681004524231, 0.20353291928768158, 0.0)),
    Vector((-0.22390709817409515, -0.9746105074882507, 0.0)),
    Vector((-0.9757033586502075, 0.2190958708524704, 0.0)),
    Vector((-0.45788809657096863, 0.8890097737312317, 0.0)),
    Vector((-0.8292457461357117, -0.5588840842247009, 0.0)),
    Vector((0.5127314329147339, -0.8585489988327026, 0.0)),
    Vector((-0.22485677897930145, -0.9743918776512146, 0.0)),
    Vector((-0.9797431826591492, 0.2002580761909485, 0.0)),
    Vector((-0.2691951096057892, -0.9630856513977051, 0.0)),
    Vector((-0.9749736785888672, 0.2223205864429474, 0.0)),
    Vector((0.18982049822807312, 0.9818188548088074, 0.0)),
    Vector((-0.9821337461471558, 0.18818409740924835, 0.0)),
    Vector((-0.1854260414838791, -0.9826582670211792, 0.0)),
    Vector((-0.9794389009475708, 0.20174118876457214, 0.0)),
    Vector((-0.5252562165260315, 0.8509441018104553, 0.0)),
    Vector((-0.8340609669685364, -0.5516724586486816, 0.0))
]
holesInfo = None
firstVertIndex = 135
numPolygonVerts = 135
faces = []

bpypolyskel.debug_outputs["skeleton"] = 1


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
    assert not bpypolyskel.check_edge_crossing(bpypolyskel.debug_outputs["skeleton"])
