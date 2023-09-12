import pytest
from mathutils import Vector

from bpypolyskel import bpypolyskel

verts = [
    Vector((1.121612787246704, 1.1354589462280273, 0.0)),
    Vector((0.23155872523784637, 2.048278570175171, 0.0)),
    Vector((-0.7670384645462036, 0.9573476910591125, 0.0)),
    Vector((-4.9423298835754395, 3.7403371334075928, 0.0)),
    Vector((-4.377904415130615, 5.232017993927002, 0.0)),
    Vector((-5.513988494873047, 5.577109336853027, 0.0)),
    Vector((-5.962635040283203, 4.1744842529296875, 0.0)),
    Vector((-11.122050285339355, 4.597506523132324, 0.0)),
    Vector((-11.230588912963867, 6.400882244110107, 0.0)),
    Vector((-12.482453346252441, 6.24503755569458, 0.0)),
    Vector((-12.330496788024902, 4.6754326820373535, 0.0)),
    Vector((-16.5998592376709, 4.230165958404541, 0.0)),
    Vector((-16.759050369262695, 5.933354377746582, 0.0)),
    Vector((-17.80106544494629, 5.855434417724609, 0.0)),
    Vector((-17.670818328857422, 4.118849754333496, 0.0)),
    Vector((-22.714458465576172, 3.5622708797454834, 0.0)),
    Vector((-23.220972061157227, 8.50485897064209, 0.0)),
    Vector((-31.774166107177734, 7.6254777908325195, 0.0)),
    Vector((-31.26766014099121, 2.660625457763672, 0.0)),
    Vector((-36.007381439208984, 2.170848846435547, 0.0)),
    Vector((-36.18827438354492, 3.8740382194519043, 0.0)),
    Vector((-37.29541778564453, 3.751594305038452, 0.0)),
    Vector((-37.12175750732422, 2.0706686973571777, 0.0)),
    Vector((-41.15956497192383, 1.658815622329712, 0.0)),
    Vector((-41.268096923828125, 3.261817216873169, 0.0)),
    Vector((-42.33905792236328, 3.195033550262451, 0.0)),
    Vector((-42.25223159790039, 2.0150463581085205, 0.0)),
    Vector((-47.15838623046875, 1.7256559133529663, 0.0)),
    Vector((-47.237972259521484, 2.8499834537506104, 0.0)),
    Vector((-48.3668212890625, 2.7386739253997803, 0.0)),
    Vector((-48.294471740722656, 1.7034019231796265, 0.0)),
    Vector((-52.628963470458984, 1.3583515882492065, 0.0)),
    Vector((-52.67960739135742, 2.4604151248931885, 0.0)),
    Vector((-53.98936080932617, 2.3379764556884766, 0.0)),
    Vector((-53.960426330566406, 1.2470451593399048, 0.0)),
    Vector((-58.30215835571289, 0.8352077007293701, 0.0)),
    Vector((-58.36003875732422, 1.8816114664077759, 0.0)),
    Vector((-63.40367889404297, 1.5365773439407349, 0.0)),
    Vector((-63.3530387878418, 0.4901735782623291, 0.0)),
    Vector((-66.34883117675781, 0.27870213985443115, 0.0)),
    Vector((-66.42118072509766, 1.1915228366851807, 0.0)),
    Vector((-67.70199584960938, 1.0802191495895386, 0.0)),
    Vector((-67.65135192871094, 0.33437788486480713, 0.0)),
    Vector((-67.97697448730469, 0.5681528449058533, 0.0)),
    Vector((-68.38944244384766, 0.6572136282920837, 0.0)),
    Vector((-68.85979461669922, 0.6683514714241028, 0.0)),
    Vector((-69.34461975097656, 0.512510359287262, 0.0)),
    Vector((-69.73538208007812, 0.17855684459209442, 0.0)),
    Vector((-70.02483367919922, -0.1999257206916809, 0.0)),
    Vector((-70.19127655029297, -0.7119932174682617, 0.0)),
    Vector((-70.2129898071289, -1.1906667947769165, 0.0)),
    Vector((-70.05379486083984, -1.6136828660964966, 0.0)),
    Vector((-70.27088928222656, -1.7806593179702759, 0.0)),
    Vector((-74.37382507324219, -2.103431463241577, 0.0)),
    Vector((-74.41722869873047, -1.3353264331817627, 0.0)),
    Vector((-75.58226776123047, -1.413233995437622, 0.0)),
    Vector((-75.07583618164062, -9.383716583251953, 0.0)),
    Vector((-77.0440902709961, -9.48387622833252, 0.0)),
    Vector((-79.66365051269531, -12.355881690979004, 0.0)),
    Vector((-79.48280334472656, -16.240934371948242, 0.0)),
    Vector((-76.61005401611328, -18.856983184814453, 0.0)),
    Vector((-74.5187759399414, -18.76795768737793, 0.0)),
    Vector((-74.05577087402344, -27.03900146484375, 0.0)),
    Vector((-72.73876953125, -26.961095809936523, 0.0)),
    Vector((-72.79664611816406, -25.502809524536133, 0.0)),
    Vector((-69.85872650146484, -25.33586883544922, 0.0)),
    Vector((-69.72126007080078, -27.484336853027344, 0.0)),
    Vector((-67.81813049316406, -27.306249618530273, 0.0)),
    Vector((-67.7891845703125, -27.3173828125, 0.0)),
    Vector((-67.9483871459961, -27.762657165527344, 0.0)),
    Vector((-67.95562744140625, -28.330387115478516, 0.0)),
    Vector((-67.81090545654297, -28.675479888916016, 0.0)),
    Vector((-67.47080993652344, -29.053970336914062, 0.0)),
    Vector((-67.09452819824219, -29.321142196655273, 0.0)),
    Vector((-66.63864135742188, -29.432466506958008, 0.0)),
    Vector((-66.26235961914062, -29.387943267822266, 0.0)),
    Vector((-65.76305389404297, -29.09851837158203, 0.0)),
    Vector((-65.39400482177734, -28.742300033569336, 0.0)),
    Vector((-65.33611297607422, -29.087390899658203, 0.0)),
    Vector((-64.56183624267578, -29.031742095947266, 0.0)),
    Vector((-64.65589904785156, -28.118919372558594, 0.0)),
    Vector((-61.75415802001953, -27.89631462097168, 0.0)),
    Vector((-61.68180465698242, -28.88705825805664, 0.0)),
    Vector((-56.73944091796875, -28.519758224487305, 0.0)),
    Vector((-56.81903076171875, -27.584674835205078, 0.0)),
    Vector((-51.659584045410156, -27.27303123474121, 0.0)),
    Vector((-51.572757720947266, -28.230379104614258, 0.0)),
    Vector((-50.45113754272461, -28.1524658203125, 0.0)),
    Vector((-50.53072738647461, -27.183984756469727, 0.0)),
    Vector((-46.217918395996094, -26.83893394470215, 0.0)),
    Vector((-46.138328552246094, -27.818546295166016, 0.0)),
    Vector((-45.002235412597656, -27.72949981689453, 0.0)),
    Vector((-45.0963020324707, -26.749887466430664, 0.0)),
    Vector((-40.276954650878906, -26.427099227905273, 0.0)),
    Vector((-40.13947677612305, -27.59595489501953, 0.0)),
    Vector((-39.0902214050293, -27.506906509399414, 0.0)),
    Vector((-39.25664138793945, -25.915037155151367, 0.0)),
    Vector((-34.32875442504883, -25.402999877929688, 0.0)),
    Vector((-34.155094146728516, -27.05052947998047, 0.0)),
    Vector((-33.06241989135742, -26.96148109436035, 0.0)),
    Vector((-33.236080169677734, -25.280555725097656, 0.0)),
    Vector((-28.51804542541504, -24.779644012451172, 0.0)),
    Vector((-27.997060775756836, -29.71109962463379, 0.0)),
    Vector((-19.45827865600586, -28.83171272277832, 0.0)),
    Vector((-19.957563400268555, -23.900259017944336, 0.0)),
    Vector((-14.819828033447266, -23.377073287963867, 0.0)),
    Vector((-14.646162033081055, -25.057998657226562, 0.0)),
    Vector((-13.596906661987305, -24.935548782348633, 0.0)),
    Vector((-13.7705717086792, -23.28801918029785, 0.0)),
    Vector((-8.965704917907715, -22.809356689453125, 0.0)),
    Vector((-8.784801483154297, -24.334434509277344, 0.0)),
    Vector((-7.6342387199401855, -24.1785888671875, 0.0)),
    Vector((-7.815142631530762, -22.65351104736328, 0.0)),
    Vector((-3.046457290649414, -20.950326919555664, 0.0)),
    Vector((-2.250471353530884, -22.252765655517578, 0.0)),
    Vector((-1.2446335554122925, -21.607112884521484, 0.0)),
    Vector((-2.047855854034424, -20.293542861938477, 0.0)),
    Vector((1.2012150287628174, -16.597736358642578, 0.0)),
    Vector((2.4386115074157715, -17.521686553955078, 0.0)),
    Vector((3.0536911487579346, -16.475284576416016, 0.0)),
    Vector((1.8018221855163574, -15.606992721557617, 0.0)),
    Vector((3.292483329772949, -10.820253372192383, 0.0)),
    Vector((4.638422012329102, -11.020627975463867, 0.0)),
    Vector((4.754200458526611, -9.818376541137695, 0.0)),
    Vector((3.335900068283081, -9.606870651245117, 0.0)),
    Vector((2.858306884765625, -5.198619365692139, 0.0)),
    Vector((4.131881237030029, -4.675416946411133, 0.0)),
    Vector((3.66876220703125, -3.3729794025421143, 0.0)),
    Vector((2.438605546951294, -3.929577589035034, 0.0)),
    Vector((0.0, 0.0, 0.0)),
    Vector((1.121612787246704, 1.1354589462280273, 8.0)),
    Vector((0.23155872523784637, 2.048278570175171, 8.0)),
    Vector((-0.7670384645462036, 0.9573476910591125, 8.0)),
    Vector((-4.9423298835754395, 3.7403371334075928, 8.0)),
    Vector((-4.377904415130615, 5.232017993927002, 8.0)),
    Vector((-5.513988494873047, 5.577109336853027, 8.0)),
    Vector((-5.962635040283203, 4.1744842529296875, 8.0)),
    Vector((-11.122050285339355, 4.597506523132324, 8.0)),
    Vector((-11.230588912963867, 6.400882244110107, 8.0)),
    Vector((-12.482453346252441, 6.24503755569458, 8.0)),
    Vector((-12.330496788024902, 4.6754326820373535, 8.0)),
    Vector((-16.5998592376709, 4.230165958404541, 8.0)),
    Vector((-16.759050369262695, 5.933354377746582, 8.0)),
    Vector((-17.80106544494629, 5.855434417724609, 8.0)),
    Vector((-17.670818328857422, 4.118849754333496, 8.0)),
    Vector((-22.714458465576172, 3.5622708797454834, 8.0)),
    Vector((-23.220972061157227, 8.50485897064209, 8.0)),
    Vector((-31.774166107177734, 7.6254777908325195, 8.0)),
    Vector((-31.26766014099121, 2.660625457763672, 8.0)),
    Vector((-36.007381439208984, 2.170848846435547, 8.0)),
    Vector((-36.18827438354492, 3.8740382194519043, 8.0)),
    Vector((-37.29541778564453, 3.751594305038452, 8.0)),
    Vector((-37.12175750732422, 2.0706686973571777, 8.0)),
    Vector((-41.15956497192383, 1.658815622329712, 8.0)),
    Vector((-41.268096923828125, 3.261817216873169, 8.0)),
    Vector((-42.33905792236328, 3.195033550262451, 8.0)),
    Vector((-42.25223159790039, 2.0150463581085205, 8.0)),
    Vector((-47.15838623046875, 1.7256559133529663, 8.0)),
    Vector((-47.237972259521484, 2.8499834537506104, 8.0)),
    Vector((-48.3668212890625, 2.7386739253997803, 8.0)),
    Vector((-48.294471740722656, 1.7034019231796265, 8.0)),
    Vector((-52.628963470458984, 1.3583515882492065, 8.0)),
    Vector((-52.67960739135742, 2.4604151248931885, 8.0)),
    Vector((-53.98936080932617, 2.3379764556884766, 8.0)),
    Vector((-53.960426330566406, 1.2470451593399048, 8.0)),
    Vector((-58.30215835571289, 0.8352077007293701, 8.0)),
    Vector((-58.36003875732422, 1.8816114664077759, 8.0)),
    Vector((-63.40367889404297, 1.5365773439407349, 8.0)),
    Vector((-63.3530387878418, 0.4901735782623291, 8.0)),
    Vector((-66.34883117675781, 0.27870213985443115, 8.0)),
    Vector((-66.42118072509766, 1.1915228366851807, 8.0)),
    Vector((-67.70199584960938, 1.0802191495895386, 8.0)),
    Vector((-67.65135192871094, 0.33437788486480713, 8.0)),
    Vector((-67.97697448730469, 0.5681528449058533, 8.0)),
    Vector((-68.38944244384766, 0.6572136282920837, 8.0)),
    Vector((-68.85979461669922, 0.6683514714241028, 8.0)),
    Vector((-69.34461975097656, 0.512510359287262, 8.0)),
    Vector((-69.73538208007812, 0.17855684459209442, 8.0)),
    Vector((-70.02483367919922, -0.1999257206916809, 8.0)),
    Vector((-70.19127655029297, -0.7119932174682617, 8.0)),
    Vector((-70.2129898071289, -1.1906667947769165, 8.0)),
    Vector((-70.05379486083984, -1.6136828660964966, 8.0)),
    Vector((-70.27088928222656, -1.7806593179702759, 8.0)),
    Vector((-74.37382507324219, -2.103431463241577, 8.0)),
    Vector((-74.41722869873047, -1.3353264331817627, 8.0)),
    Vector((-75.58226776123047, -1.413233995437622, 8.0)),
    Vector((-75.07583618164062, -9.383716583251953, 8.0)),
    Vector((-77.0440902709961, -9.48387622833252, 8.0)),
    Vector((-79.66365051269531, -12.355881690979004, 8.0)),
    Vector((-79.48280334472656, -16.240934371948242, 8.0)),
    Vector((-76.61005401611328, -18.856983184814453, 8.0)),
    Vector((-74.5187759399414, -18.76795768737793, 8.0)),
    Vector((-74.05577087402344, -27.03900146484375, 8.0)),
    Vector((-72.73876953125, -26.961095809936523, 8.0)),
    Vector((-72.79664611816406, -25.502809524536133, 8.0)),
    Vector((-69.85872650146484, -25.33586883544922, 8.0)),
    Vector((-69.72126007080078, -27.484336853027344, 8.0)),
    Vector((-67.81813049316406, -27.306249618530273, 8.0)),
    Vector((-67.7891845703125, -27.3173828125, 8.0)),
    Vector((-67.9483871459961, -27.762657165527344, 8.0)),
    Vector((-67.95562744140625, -28.330387115478516, 8.0)),
    Vector((-67.81090545654297, -28.675479888916016, 8.0)),
    Vector((-67.47080993652344, -29.053970336914062, 8.0)),
    Vector((-67.09452819824219, -29.321142196655273, 8.0)),
    Vector((-66.63864135742188, -29.432466506958008, 8.0)),
    Vector((-66.26235961914062, -29.387943267822266, 8.0)),
    Vector((-65.76305389404297, -29.09851837158203, 8.0)),
    Vector((-65.39400482177734, -28.742300033569336, 8.0)),
    Vector((-65.33611297607422, -29.087390899658203, 8.0)),
    Vector((-64.56183624267578, -29.031742095947266, 8.0)),
    Vector((-64.65589904785156, -28.118919372558594, 8.0)),
    Vector((-61.75415802001953, -27.89631462097168, 8.0)),
    Vector((-61.68180465698242, -28.88705825805664, 8.0)),
    Vector((-56.73944091796875, -28.519758224487305, 8.0)),
    Vector((-56.81903076171875, -27.584674835205078, 8.0)),
    Vector((-51.659584045410156, -27.27303123474121, 8.0)),
    Vector((-51.572757720947266, -28.230379104614258, 8.0)),
    Vector((-50.45113754272461, -28.1524658203125, 8.0)),
    Vector((-50.53072738647461, -27.183984756469727, 8.0)),
    Vector((-46.217918395996094, -26.83893394470215, 8.0)),
    Vector((-46.138328552246094, -27.818546295166016, 8.0)),
    Vector((-45.002235412597656, -27.72949981689453, 8.0)),
    Vector((-45.0963020324707, -26.749887466430664, 8.0)),
    Vector((-40.276954650878906, -26.427099227905273, 8.0)),
    Vector((-40.13947677612305, -27.59595489501953, 8.0)),
    Vector((-39.0902214050293, -27.506906509399414, 8.0)),
    Vector((-39.25664138793945, -25.915037155151367, 8.0)),
    Vector((-34.32875442504883, -25.402999877929688, 8.0)),
    Vector((-34.155094146728516, -27.05052947998047, 8.0)),
    Vector((-33.06241989135742, -26.96148109436035, 8.0)),
    Vector((-33.236080169677734, -25.280555725097656, 8.0)),
    Vector((-28.51804542541504, -24.779644012451172, 8.0)),
    Vector((-27.997060775756836, -29.71109962463379, 8.0)),
    Vector((-19.45827865600586, -28.83171272277832, 8.0)),
    Vector((-19.957563400268555, -23.900259017944336, 8.0)),
    Vector((-14.819828033447266, -23.377073287963867, 8.0)),
    Vector((-14.646162033081055, -25.057998657226562, 8.0)),
    Vector((-13.596906661987305, -24.935548782348633, 8.0)),
    Vector((-13.7705717086792, -23.28801918029785, 8.0)),
    Vector((-8.965704917907715, -22.809356689453125, 8.0)),
    Vector((-8.784801483154297, -24.334434509277344, 8.0)),
    Vector((-7.6342387199401855, -24.1785888671875, 8.0)),
    Vector((-7.815142631530762, -22.65351104736328, 8.0)),
    Vector((-3.046457290649414, -20.950326919555664, 8.0)),
    Vector((-2.250471353530884, -22.252765655517578, 8.0)),
    Vector((-1.2446335554122925, -21.607112884521484, 8.0)),
    Vector((-2.047855854034424, -20.293542861938477, 8.0)),
    Vector((1.2012150287628174, -16.597736358642578, 8.0)),
    Vector((2.4386115074157715, -17.521686553955078, 8.0)),
    Vector((3.0536911487579346, -16.475284576416016, 8.0)),
    Vector((1.8018221855163574, -15.606992721557617, 8.0)),
    Vector((3.292483329772949, -10.820253372192383, 8.0)),
    Vector((4.638422012329102, -11.020627975463867, 8.0)),
    Vector((4.754200458526611, -9.818376541137695, 8.0)),
    Vector((3.335900068283081, -9.606870651245117, 8.0)),
    Vector((2.858306884765625, -5.198619365692139, 8.0)),
    Vector((4.131881237030029, -4.675416946411133, 8.0)),
    Vector((3.66876220703125, -3.3729794025421143, 8.0)),
    Vector((2.438605546951294, -3.929577589035034, 8.0)),
    Vector((0.0, 0.0, 8.0)),
]
unitVectors = [
    Vector((-0.6981222033500671, 0.7159786224365234, 0.0)),
    Vector((-0.6752017140388489, -0.7376331686973572, 0.0)),
    Vector((-0.8320997953414917, 0.554625928401947, 0.0)),
    Vector((0.3538952171802521, 0.9352850914001465, 0.0)),
    Vector((-0.9568316340446472, 0.29064249992370605, 0.0)),
    Vector((-0.3046565353870392, -0.9524621963500977, 0.0)),
    Vector((-0.9966555833816528, 0.0817161425948143, 0.0)),
    Vector((-0.06007764860987663, 0.9981937408447266, 0.0)),
    Vector((-0.992340087890625, -0.12353648245334625, 0.0)),
    Vector((0.09636145085096359, -0.9953463673591614, 0.0)),
    Vector((-0.9946054220199585, -0.10373087227344513, 0.0)),
    Vector((-0.0930609405040741, 0.9956604242324829, 0.0)),
    Vector((-0.997215747833252, -0.0745699480175972, 0.0)),
    Vector((0.0747918114066124, -0.9971991181373596, 0.0)),
    Vector((-0.993966281414032, -0.10968677699565887, 0.0)),
    Vector((-0.10194551199674606, 0.9947900176048279, 0.0)),
    Vector((-0.994756281375885, -0.10227406769990921, 0.0)),
    Vector((0.1014915481209755, -0.9948363900184631, 0.0)),
    Vector((-0.9947034120559692, -0.10278715193271637, 0.0)),
    Vector((-0.10561434924602509, 0.9944071769714355, 0.0)),
    Vector((-0.993939995765686, -0.10992424190044403, 0.0)),
    Vector((0.10276531428098679, -0.9947056174278259, 0.0)),
    Vector((-0.9948383569717407, -0.10147270560264587, 0.0)),
    Vector((-0.06755080074071884, 0.9977158308029175, 0.0)),
    Vector((-0.9980612397193909, -0.06223773956298828, 0.0)),
    Vector((0.07338403165340424, -0.9973037242889404, 0.0)),
    Vector((-0.9982649683952332, -0.05888284370303154, 0.0)),
    Vector((-0.07060877978801727, 0.9975041151046753, 0.0)),
    Vector((-0.9951737523078918, -0.0981285497546196, 0.0)),
    Vector((0.06971454620361328, -0.9975669980049133, 0.0)),
    Vector((-0.9968464374542236, -0.0793546810746193, 0.0)),
    Vector((-0.045905277132987976, 0.9989458322525024, 0.0)),
    Vector((-0.9956589937210083, -0.0930764228105545, 0.0)),
    Vector((0.02651340886950493, -0.9996485114097595, 0.0)),
    Vector((-0.9955313801765442, -0.09443169087171555, 0.0)),
    Vector((-0.05522921681404114, 0.9984737038612366, 0.0)),
    Vector((-0.9976683259010315, -0.06825023144483566, 0.0)),
    Vector((0.04833785071969032, -0.998831033706665, 0.0)),
    Vector((-0.9975178241729736, -0.07041426748037338, 0.0)),
    Vector((-0.07901152968406677, 0.996873676776886, 0.0)),
    Vector((-0.9962453842163086, -0.08657438308000565, 0.0)),
    Vector((0.06774574518203735, -0.9977025985717773, 0.0)),
    Vector((-0.8123301267623901, 0.583198070526123, 0.0)),
    Vector((-0.9774735569953918, 0.21105776727199554, 0.0)),
    Vector((-0.9997197389602661, 0.023673158138990402, 0.0)),
    Vector((-0.9520260095596313, -0.306017130613327, 0.0)),
    Vector((-0.7602032423019409, -0.649685263633728, 0.0)),
    Vector((-0.6074815392494202, -0.7943337559700012, 0.0)),
    Vector((-0.3091212511062622, -0.9510226249694824, 0.0)),
    Vector((-0.04531470686197281, -0.9989727735519409, 0.0)),
    Vector((0.35221704840660095, -0.9359183311462402, 0.0)),
    Vector((-0.7926579713821411, -0.6096665859222412, 0.0)),
    Vector((-0.9969199895858765, -0.07842627912759781, 0.0)),
    Vector((-0.05641740933060646, 0.9984073042869568, 0.0)),
    Vector((-0.9977716207504272, -0.06672218441963196, 0.0)),
    Vector((0.06341051310300827, -0.9979875087738037, 0.0)),
    Vector((-0.99870765209198, -0.05082179605960846, 0.0)),
    Vector((-0.673889696598053, -0.7388319969177246, 0.0)),
    Vector((0.04649912565946579, -0.9989182949066162, 0.0)),
    Vector((0.7393686175346375, -0.6733008027076721, 0.0)),
    Vector((0.9990951418876648, 0.04253137856721878, 0.0)),
    Vector((0.05589153617620468, -0.9984368681907654, 0.0)),
    Vector((0.9982550144195557, 0.05905059352517128, 0.0)),
    Vector((-0.039656862616539, 0.9992133975028992, 0.0)),
    Vector((0.9983896017074585, 0.056731246411800385, 0.0)),
    Vector((0.06385289877653122, -0.997959315776825, 0.0)),
    Vector((0.9956502914428711, 0.0931689664721489, 0.0)),
    Vector((0.9333440065383911, -0.35898318886756897, 0.0)),
    Vector((-0.3366664946079254, -0.9416239261627197, 0.0)),
    Vector((-0.012752024456858635, -0.9999186396598816, 0.0)),
    Vector((0.3867395520210266, -0.9221889972686768, 0.0)),
    Vector((0.6683717966079712, -0.7438273429870605, 0.0)),
    Vector((0.8153708577156067, -0.5789389610290527, 0.0)),
    Vector((0.9714553952217102, -0.23722247779369354, 0.0)),
    Vector((0.9930723309516907, 0.11750449985265732, 0.0)),
    Vector((0.8651607632637024, 0.5014944672584534, 0.0)),
    Vector((0.71950364112854, 0.6944886445999146, 0.0)),
    Vector((0.16544635593891144, -0.9862187504768372, 0.0)),
    Vector((0.9974271655082703, 0.07168706506490707, 0.0)),
    Vector((-0.10250329971313477, 0.9947326183319092, 0.0)),
    Vector((0.9970703125, 0.07648945599794388, 0.0)),
    Vector((0.07283537834882736, -0.9973439574241638, 0.0)),
    Vector((0.997249960899353, 0.07411229610443115, 0.0)),
    Vector((-0.0848085880279541, 0.9963972568511963, 0.0)),
    Vector((0.9981808066368103, 0.06029263883829117, 0.0)),
    Vector((0.09032392501831055, -0.9959124326705933, 0.0)),
    Vector((0.9975960850715637, 0.06929795444011688, 0.0)),
    Vector((-0.08190396428108215, 0.9966402649879456, 0.0)),
    Vector((0.9968147873878479, 0.07975121587514877, 0.0)),
    Vector((0.08097943663597107, -0.996715784072876, 0.0)),
    Vector((0.996942400932312, 0.07813990116119385, 0.0)),
    Vector((-0.09558466076850891, 0.995421290397644, 0.0)),
    Vector((0.9977645874023438, 0.06682785600423813, 0.0)),
    Vector((0.11681228876113892, -0.9931540489196777, 0.0)),
    Vector((0.9964180588722229, 0.08456417918205261, 0.0)),
    Vector((-0.10397708415985107, 0.9945797324180603, 0.0)),
    Vector((0.9946451187133789, 0.10334964841604233, 0.0)),
    Vector((0.10482574999332428, -0.9944906234741211, 0.0)),
    Vector((0.9966956973075867, 0.08122653514146805, 0.0)),
    Vector((-0.10276533663272858, 0.9947056770324707, 0.0)),
    Vector((0.9944111704826355, 0.10557620227336884, 0.0)),
    Vector((0.10506054759025574, -0.9944658279418945, 0.0)),
    Vector((0.9947386384010315, 0.10244553536176682, 0.0)),
    Vector((-0.10072999447584152, 0.9949138760566711, 0.0)),
    Vector((0.9948551058769226, 0.10130806267261505, 0.0)),
    Vector((0.1027686819434166, -0.9947052597999573, 0.0)),
    Vector((0.993259072303772, 0.11591500788927078, 0.0)),
    Vector((-0.10482858121395111, 0.9944902658462524, 0.0)),
    Vector((0.9950745701789856, 0.09912966936826706, 0.0)),
    Vector((0.11779333651065826, -0.9930381178855896, 0.0)),
    Vector((0.9909507036209106, 0.13422591984272003, 0.0)),
    Vector((-0.11779364943504333, 0.9930381178855896, 0.0)),
    Vector((0.9417368173599243, 0.33635079860687256, 0.0)),
    Vector((0.5214744210243225, -0.8532669544219971, 0.0)),
    Vector((0.8415426015853882, 0.5401908159255981, 0.0)),
    Vector((-0.5216794013977051, 0.8531416654586792, 0.0)),
    Vector((0.6602569222450256, 0.7510398626327515, 0.0)),
    Vector((0.8012717366218567, -0.5983006954193115, 0.0)),
    Vector((0.5067441463470459, 0.8620966076850891, 0.0)),
    Vector((-0.8216962814331055, 0.5699256062507629, 0.0)),
    Vector((0.2973308563232422, 0.9547744989395142, 0.0)),
    Vector((0.9890992045402527, -0.14725066721439362, 0.0)),
    Vector((0.0958578959107399, 0.9953950643539429, 0.0)),
    Vector((-0.9890627861022949, 0.14749526977539062, 0.0)),
    Vector((-0.10771044343709946, 0.9941823482513428, 0.0)),
    Vector((0.9249873161315918, 0.37999793887138367, 0.0)),
    Vector((-0.3350290060043335, 0.9422077536582947, 0.0)),
    Vector((-0.9110804796218872, -0.4122285842895508, 0.0)),
    Vector((-0.5272938013076782, 0.8496831059455872, 0.0)),
    Vector((0.7027557492256165, 0.7114311456680298, 0.0)),
]
holesInfo = None
firstVertIndex = 130
numPolygonVerts = 130
faces = []

bpypolyskel.debug_outputs["skeleton"] = 1


@pytest.mark.dependency()
@pytest.mark.timeout(10)
def test_polygonize():
    global faces
    faces = bpypolyskel.polygonize(
        verts, firstVertIndex, numPolygonVerts, holesInfo, 0.0, 0.5, None, unitVectors
    )


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
