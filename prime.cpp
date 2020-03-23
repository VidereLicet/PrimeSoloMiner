/*******************************************************************************************
 
			Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++
   
 [Learn, Create, but do not Forge] Viz. http://www.opensource.org/licenses/mit-license.php
  
*******************************************************************************************/

#include "core.h"
#include "cuda/cuda.h"

// use e.g. latest primesieve 0.5.4 package
#include <primesieve.hpp>

// used for implementing a work queue to submit work for CPU verification
#include <boost/thread/thread.hpp>

#include <queue>
#include <atomic>
#include <algorithm>
#include <inttypes.h>

using namespace std;

unsigned int *primes;
unsigned int *inverses;
uint64_t *invK;

extern unsigned int nThreadsCPU;

unsigned int nBitArray_Size[8] = { 0 };
mpz_t  zPrimorial;

const unsigned int nPrimorialEndPrime = 12;

unsigned int nPrimeLimitA[8] = { 0 };
unsigned int nPrimeLimitB[8] = { 0 };
unsigned int nPrimeLimit = 0;

unsigned int nSharedSizeKB[8] = { 48 };
unsigned int nThreadsKernelA[8] = { 768 };

unsigned int nFourChainsFoundCounter = 0;
unsigned int nFiveChainsFoundCounter = 0;
unsigned int nSixChainsFoundCounter = 0;
unsigned int nSevenChainsFoundCounter = 0;
unsigned int nEightChainsFoundCounter = 0;

extern volatile unsigned int nBestHeight;
extern std::atomic<uint64_t> SievedBits;
extern std::atomic<uint64_t> CheckedCandidates;
extern std::atomic<uint64_t> PrimesFound;
extern std::atomic<uint64_t> PrimesChecked;

uint64_t originTable[1024] = {
	7312421ULL, 45420491ULL, 111456461ULL, 135780761ULL, 163528481ULL, 185990921ULL, 193978901ULL,
	201306221ULL, 205390301ULL, 218993891ULL, 231426311ULL, 268513361ULL, 288603431ULL, 303738551ULL,
	318723521ULL, 334549331ULL, 347822591ULL, 409083791ULL, 417071771ULL, 424399091ULL, 454519181ULL,
	488032661ULL, 511696301ULL, 529383971ULL, 541816391ULL, 552867431ULL, 557642201ULL, 559504061ULL,
	570915461ULL, 594729251ULL, 597281801ULL, 608002511ULL, 629113601ULL, 632176661ULL, 644939411ULL,
	662627081ULL, 664338791ULL, 676591031ULL, 682026461ULL, 700074491ULL, 711125531ULL, 734789171ULL,
	737161541ULL, 764909261ULL, 806771081ULL, 852206471ULL, 866320571ULL, 868032281ULL, 869894141ULL,
	905119331ULL, 937791971ULL, 1029863951ULL, 1055899961ULL, 1062866921ULL, 1066951001ULL, 1073587631ULL,
	1075299341ULL, 1091125151ULL, 1159022981ULL, 1191695621ULL, 1196110031ULL, 1198662581ULL, 1207521431ULL,
	1251245111ULL, 1277971811ULL, 1297371191ULL, 1298392211ULL, 1331905691ULL, 1406440151ULL, 1421755451ULL,
	1434187871ULL, 1451875541ULL, 1520464061ULL, 1532536121ULL, 1539172751ULL, 1554998561ULL, 1592445971ULL,
	1629533021ULL, 1644848321ULL, 1650644111ULL, 1674968411ULL, 1692505931ULL, 1710193601ULL, 1732145531ULL,
	1747641011ULL, 1762265621ULL, 1778091431ULL, 1791364691ULL, 1792055381ULL, 1815538841ULL, 1852625891ULL,
	1865388641ULL, 1897040261ULL, 1947250421ULL, 1955238401ULL, 1985358491ULL, 2001184301ULL, 2038631711ULL,
	2051394461ULL, 2072655701ULL, 2088481511ULL, 2121844841ULL, 2132565551ULL, 2143616591ULL, 2170343291ULL,
	2206589501ULL, 2208451361ULL, 2213226131ULL, 2250313181ULL, 2274487331ULL, 2276349191ULL, 2295748571ULL,
	2311574381ULL, 2361784541ULL, 2390883611ULL, 2393436161ULL, 2427970661ULL, 2436319001ULL, 2497580201ULL,
	2498421041ULL, 2500132751ULL, 2517820421ULL, 2518841441ULL, 2521904501ULL, 2566318871ULL, 2584877411ULL,
	2591514041ULL, 2602565081ULL, 2640012491ULL, 2659411871ULL, 2672174621ULL, 2684426861ULL, 2733946331ULL,
	2740913291ULL, 2769171521ULL, 2807970281ULL, 2812895201ULL, 2847609881ULL, 2882504741ULL, 2895267491ULL,
	2912955161ULL, 2976438581ULL, 2980852991ULL, 3014216321ULL, 3024937031ULL, 3035988071ULL, 3098960981ULL,
	3101513531ULL, 3116648651ULL, 3118360361ULL, 3154096061ULL, 3167699651ULL, 3188120051ULL, 3192894821ULL,
	3229981871ULL, 3259080941ULL, 3271843691ULL, 3324606401ULL, 3341453231ULL, 3342294071ULL, 3354726491ULL,
	3390792521ULL, 3403224941ULL, 3435387071ULL, 3477248891ULL, 3483885521ULL, 3494936561ULL, 3528299891ULL,
	3577819361ULL, 3582233771ULL, 3613044551ULL, 3613885391ULL, 3614906411ULL, 3656768231ULL, 3694215641ULL,
	3718029431ULL, 3755476841ULL, 3794275601ULL, 3805326641ULL, 3836978261ULL, 3860461721ULL, 3862173431ULL,
	3868810061ULL, 3879861101ULL, 3904876091ULL, 3906587801ULL, 3954395561ULL, 3997969091ULL, 4028419511ULL,
	4102953971ULL, 4115716721ULL, 4140401381ULL, 4153164131ULL, 4175626571ULL, 4177488431ULL, 4179200141ULL,
	4197578501ULL, 4198599521ULL, 4214425331ULL, 4216287191ULL, 4234665551ULL, 4245386261ULL, 4319410211ULL,
	4327758551ULL, 4338809591ULL, 4345446221ULL, 4413344051ULL, 4420671371ULL, 4456917581ULL, 4463554211ULL,
	4507968581ULL, 4538088671ULL, 4545055631ULL, 4553914481ULL, 4561902461ULL, 4575175721ULL, 4599349871ULL,
	4636436921ULL, 4643764241ULL, 4686647081ULL, 4704334751ULL, 4715746151ULL, 4731061451ULL, 4741782161ULL,
	4748749121ULL, 4761181541ULL, 4772232581ULL, 4777007351ULL, 4798268591ULL, 4815806111ULL, 4886256491ULL,
	4909739951ULL, 4914664871ULL, 4938839021ULL, 4952442611ULL, 4954154321ULL, 4975926071ULL, 4984274411ULL,
	4997037161ULL, 5067998051ULL, 5071571621ULL, 5089259291ULL, 5132832821ULL, 5174694641ULL, 5177247191ULL,
	5200730651ULL, 5220130031ULL, 5231181071ULL, 5237817701ULL, 5245145021ULL, 5292952781ULL, 5313042851ULL,
	5360850611ULL, 5373613361ULL, 5397787511ULL, 5418027731ULL, 5419048751ULL, 5434874561ULL, 5443222901ULL,
	5446285961ULL, 5524724321ULL, 5548207781ULL, 5559619181ULL, 5583943481ULL, 5585655191ULL, 5641120601ULL,
	5653553021ULL, 5751901271ULL, 5758537901ULL, 5774363711ULL, 5778627971ULL, 5793763091ULL, 5796315641ULL,
	5807727041ULL, 5819799101ULL, 5831210501ULL, 5864213471ULL, 5907096311ULL, 5914423631ULL, 5936195381ULL,
	5951510681ULL, 5960369531ULL, 5981630771ULL, 5997456581ULL, 6001030151ULL, 6018567671ULL, 6042891971ULL,
	6089018051ULL, 6112501511ULL, 6130189181ULL, 6137516501ULL, 6159288251ULL, 6174603551ULL, 6183462401ULL,
	6185324261ULL, 6204723641ULL, 6217486391ULL, 6220549451ULL, 6248297171ULL, 6254933801ULL, 6288447281ULL,
	6290158991ULL, 6292020851ULL, 6292711541ULL, 6353282051ULL, 6395143871ULL, 6397696421ULL, 6427816511ULL,
	6432591281ULL, 6440579261ULL, 6447906581ULL, 6478026671ULL, 6581149691ULL, 6618236741ULL, 6633221711ULL,
	6640188671ULL, 6644272751ULL, 6651600071ULL, 6663672131ULL, 6701119541ULL, 6758296661ULL, 6780068411ULL,
	6784843181ULL, 6788416751ULL, 6804242561ULL, 6886765001ULL, 6891539771ULL, 6894092321ULL, 6923191391ULL,
	6980698841ULL, 6981389531ULL, 6992110241ULL, 6999077201ULL, 7011509621ULL, 7022560661ULL, 7027335431ULL,
	7029197291ULL, 7066975031ULL, 7101869891ULL, 7169767721ULL, 7180818761ULL, 7204482401ULL, 7234602491ULL,
	7239016901ULL, 7251269141ULL, 7263701561ULL, 7336013801ULL, 7357965731ULL, 7403911631ULL, 7407485201ULL,
	7427575271ULL, 7451058731ULL, 7486794431ULL, 7532560151ULL, 7544992571ULL, 7560818381ULL, 7627004501ULL,
	7628716211ULL, 7629557051ULL, 7668355811ULL, 7696614041ULL, 7698475901ULL, 7720938341ULL, 7722650051ULL,
	7768085441ULL, 7801598921ULL, 7839046331ULL, 7854361631ULL, 7903881101ULL, 7921568771ULL, 7945742921ULL,
	8008865981ULL, 8024691791ULL, 8062139201ULL, 8088175211ULL, 8107574591ULL, 8114541551ULL, 8123400401ULL,
	8143640621ULL, 8162199161ULL, 8179886831ULL, 8194871801ULL, 8201838761ULL, 8217334241ULL, 8228385281ULL,
	8231958851ULL, 8247784661ULL, 8261748611ULL, 8297994821ULL, 8322319121ULL, 8337634421ULL, 8416943651ULL,
	8424931631ULL, 8455051721ULL, 8470877531ULL, 8473430081ULL, 8484150791ULL, 8508324941ULL, 8521087691ULL,
	8543039621ULL, 8547123701ULL, 8564811371ULL, 8566523081ULL, 8613309821ULL, 8640036521ULL, 8648024501ULL,
	8676282731ULL, 8678144591ULL, 8707243661ULL, 8742468851ULL, 8765441801ULL, 8781267611ULL, 8813940251ULL,
	8831477771ULL, 8863129391ULL, 8883549791ULL, 8906012231ULL, 8925411611ULL, 8943099281ULL, 8967273431ULL,
	8980546691ULL, 8988534671ULL, 8991597731ULL, 9023759861ULL, 9036012101ULL, 9054570641ULL, 9072258311ULL,
	9074810861ULL, 9083669711ULL, 9109705721ULL, 9129105101ULL, 9154120091ULL, 9166192151ULL, 9277663511ULL,
	9282588431ULL, 9306762581ULL, 9308624441ULL, 9310336151ULL, 9317303111ULL, 9328023821ULL, 9343849631ULL,
	9352197971ULL, 9364960721ULL, 9382648391ULL, 9415321031ULL, 9439495181ULL, 9450546221ULL, 9457182851ULL,
	9494630261ULL, 9532408001ULL, 9568654211ULL, 9586341881ULL, 9588053591ULL, 9637392881ULL, 9657813281ULL,
	9728774171ULL, 9741536921ULL, 9767572931ULL, 9775921271ULL, 9786972311ULL, 9793608941ULL, 9811146461ULL,
	9823398701ULL, 9860485751ULL, 9872918171ULL, 9911716931ULL, 9916131341ULL, 9927542741ULL, 9953578751ULL,
	9997993121ULL, 10017392501ULL, 10046491571ULL, 10051927001ULL, 10083578621ULL, 10126461461ULL, 10163908871ULL,
	10187722661ULL, 10200995921ULL, 10240485371ULL, 10275019871ULL, 10304118941ULL, 10306671491ULL, 10330154951ULL,
	10331866661ULL, 10349554331ULL, 10374569321ULL, 10387001741ULL, 10467662321ULL, 10498112741ULL, 10527211811ULL,
	10572647201ULL, 10585409951ULL, 10610094611ULL, 10622857361ULL, 10659944411ULL, 10667271731ULL, 10715079491ULL,
	10721205611ULL, 10789103441ULL, 10800514841ULL, 10808502821ULL, 10845950231ULL, 10859553821ULL, 10890364601ULL,
	10926610811ULL, 10933247441ULL, 10996370501ULL, 11031595691ULL, 11043847931ULL, 11069043101ULL, 11113457471ULL,
	11147991971ULL, 11150544521ULL, 11156340311ULL, 11174027981ULL, 11200754681ULL, 11211475391ULL, 11218442351ULL,
	11220154061ULL, 11237841731ULL, 11241925811ULL, 11266940801ULL, 11279373221ULL, 11360033801ULL, 11379433181ULL,
	11384358101ULL, 11419583291ULL, 11422135841ULL, 11423847551ULL, 11453967641ULL, 11460934601ULL, 11489192831ULL,
	11491054691ULL, 11537691281ULL, 11602526051ULL, 11607450971ULL, 11631625121ULL, 11646940421ULL, 11670423881ULL,
	11696459891ULL, 11700874301ULL, 11714838251ULL, 11738321711ULL, 11751925301ULL, 11762646011ULL, 11763336701ULL,
	11818982291ULL, 11830543841ULL, 11836669961ULL, 11843306591ULL, 11880754001ULL, 11887720961ULL, 11912916131ULL,
	11915979191ULL, 11987450591ULL, 12020964071ULL, 12055348421ULL, 12066399461ULL, 12090573611ULL, 12092435471ULL,
	12103846871ULL, 12110813831ULL, 12123246251ULL, 12206969891ULL, 12221594501ULL, 12233005901ULL, 12248321201ULL,
	12266008871ULL, 12289492331ULL, 12333906701ULL, 12334927721ULL, 12339702491ULL, 12376789541ULL, 12421203911ULL,
	12430062761ULL, 12475498151ULL, 12556999571ULL, 12562795361ULL, 12580483031ULL, 12582194741ULL, 12588831371ULL,
	12599882411ULL, 12606519041ULL, 12607209731ULL, 12617930441ULL, 12644296781ULL, 12674416871ULL, 12690242681ULL,
	12691954391ULL, 12717990401ULL, 12724627031ULL, 12758140511ULL, 12762404771ULL, 12822975281ULL, 12830302601ULL,
	12852074351ULL, 12867389651ULL, 12897509741ULL, 12899221451ULL, 12917599811ULL, 12934446641ULL, 12946698881ULL,
	12947719901ULL, 12983785931ULL, 13039431521ULL, 13053395471ULL, 13075167221ULL, 13076878931ULL, 13102914941ULL,
	13133365361ULL, 13140692681ULL, 13158380351ULL, 13170812771ULL, 13207899821ULL, 13227989891ULL, 13258109981ULL,
	13273935791ULL, 13311022841ULL, 13322434241ULL, 13324296101ULL, 13324986791ULL, 13348470251ULL, 13356458231ULL,
	13363785551ULL, 13393905641ULL, 13427419121ULL, 13451082761ULL, 13461803471ULL, 13468770431ULL, 13481202851ULL,
	13492253891ULL, 13497028661ULL, 13498890521ULL, 13534115711ULL, 13536668261ULL, 13639460951ULL, 13650511991ULL,
	13654776251ULL, 13672463921ULL, 13674175631ULL, 13676548001ULL, 13695947381ULL, 13704295721ULL, 13783244591ULL,
	13791592931ULL, 13805707031ULL, 13807418741ULL, 13809280601ULL, 13827658961ULL, 13877178431ULL, 13897268501ULL,
	13920751961ULL, 13923815021ULL, 13995286421ULL, 14012974091ULL, 14033064161ULL, 14050751831ULL, 14096697731ULL,
	14098409441ULL, 14135496491ULL, 14138049041ULL, 14154895871ULL, 14166307271ULL, 14168169131ULL, 14168859821ULL,
	14273844701ULL, 14279640491ULL, 14308739561ULL, 14361141911ULL, 14373574331ULL, 14391262001ULL, 14439760451ULL,
	14471922581ULL, 14478559211ULL, 14494385021ULL, 14509369991ULL, 14513784401ULL, 14531832431ULL, 14568919481ULL,
	14584234781ULL, 14614354871ULL, 14664565031ULL, 14671531991ULL, 14687027471ULL, 14698078511ULL, 14701652081ULL,
	14717477891ULL, 14731441841ULL, 14754925301ULL, 14767688051ULL, 14770240601ULL, 14875225481ULL, 14887657901ULL,
	14894624861ULL, 14898708941ULL, 14905345571ULL, 14924744951ULL, 14940570761ULL, 14978018171ULL, 14990780921ULL,
	15012042161ULL, 15012732851ULL, 15027867971ULL, 15083003051ULL, 15109729751ULL, 15110750771ULL, 15117717731ULL,
	15145975961ULL, 15147837821ULL, 15152612591ULL, 15213873791ULL, 15215735651ULL, 15283633481ULL, 15301171001ULL,
	15332822621ULL, 15353243021ULL, 15364294061ULL, 15371621381ULL, 15436966661ULL, 15461290961ULL, 15500089721ULL,
	15504864491ULL, 15524263871ULL, 15541951541ULL, 15579398951ULL, 15611561081ULL, 15623813321ULL, 15712131551ULL,
	15730509911ULL, 15747356741ULL, 15752281661ULL, 15786996341ULL, 15821891201ULL, 15885014261ULL, 15920239451ULL,
	15953602781ULL, 15964323491ULL, 15975374531ULL, 15990689831ULL, 15999548681ULL, 16002101231ULL, 16038347441ULL,
	16056035111ULL, 16108107131ULL, 16127506511ULL, 16198467401ULL, 16211230151ULL, 16248677561ULL, 16280839691ULL,
	16330178981ULL, 16331200001ULL, 16349578361ULL, 16416635351ULL, 16423271981ULL, 16434323021ULL, 16471770431ULL,
	16516184801ULL, 16521620231ULL, 16553271851ULL, 16574382941ULL, 16600929461ULL, 16633602101ULL, 16657415891ULL,
	16707626051ULL, 16739277671ULL, 16744713101ULL, 16776364721ULL, 16799848181ULL, 16808196521ULL, 16827595901ULL,
	16843421711ULL, 16844262551ULL, 16845974261ULL, 16856694971ULL, 16867746011ULL, 16914893111ULL, 16930718921ULL,
	16937355551ULL, 16948406591ULL, 16950959141ULL, 16967805971ULL, 16985854001ULL, 17042340431ULL, 17079787841ULL,
	17092550591ULL, 17115013031ULL, 17116874891ULL, 17136964961ULL, 17174052011ULL, 17184772721ULL, 17186484431ULL,
	17190898841ULL, 17219997911ULL, 17258796671ULL, 17267145011ULL, 17284832681ULL, 17315643461ULL, 17326694501ULL,
	17340658451ULL, 17360057831ULL, 17396304041ULL, 17444802491ULL, 17447355041ULL, 17469126791ULL, 17501288921ULL,
	17538736331ULL, 17575823381ULL, 17583150701ULL, 17626033541ULL, 17643721211ULL, 17681168621ULL, 17692219661ULL,
	17700568001ULL, 17711619041ULL, 17716393811ULL, 17736634031ULL, 17787865211ULL, 17794832171ULL, 17821378691ULL,
	17829727031ULL, 17849126411ULL, 17854051331ULL, 17891829071ULL, 17928075281ULL, 17959726901ULL, 18007384511ULL,
	18010958081ULL, 18028645751ULL, 18030357461ULL, 18048045131ULL, 18063870941ULL, 18072219281ULL, 18077144201ULL,
	18116633651ULL, 18135342341ULL, 18140117111ULL, 18151168151ULL, 18159516491ULL, 18170567531ULL, 18177204161ULL,
	18182819771ULL, 18184531481ULL, 18206303231ULL, 18232339241ULL, 18252429311ULL, 18288675521ULL, 18300237071ULL,
	18306363191ULL, 18312999821ULL, 18357414191ULL, 18385672421ULL, 18487594241ULL, 18523329941ULL, 18536092691ULL,
	18573540101ULL, 18580507061ULL, 18592939481ULL, 18647564051ULL, 18665251721ULL, 18676663121ULL, 18679215671ULL,
	18691287731ULL, 18718014431ULL, 18735702101ULL, 18759185561ULL, 18770596961ULL, 18784200551ULL, 18796632971ULL,
	18803599931ULL, 18846482771ULL, 18853810091ULL, 18899755991ULL, 18921017231ULL, 18936843041ULL, 18982278431ULL,
	19019725841ULL, 19026692801ULL, 19032488591ULL, 19050176261ULL, 19051887971ULL, 19069575641ULL, 19076902961ULL,
	19122848861ULL, 19124710721ULL, 19187683631ULL, 19194320261ULL, 19227833741ULL, 19229545451ULL, 19231407311ULL,
	19232098001ULL, 19255581461ULL, 19292668511ULL, 19298284121ULL, 19337082881ULL, 19371977741ULL, 19379965721ULL,
	19387293041ULL, 19417413131ULL, 19478674331ULL, 19509124751ULL, 19520536151ULL, 19523088701ULL, 19572608171ULL,
	19579575131ULL, 19583659211ULL, 19610385911ULL, 19628073581ULL, 19638794291ULL, 19639484981ULL, 19640506001ULL,
	19724229641ULL, 19792127471ULL, 19793989331ULL, 19830926231ULL, 19833478781ULL, 19862577851ULL, 19863598871ULL,
	19868013281ULL, 19897112351ULL, 19899664901ULL, 19920775991ULL, 19931496701ULL, 19938463661ULL, 19961947121ULL,
	20006361491ULL, 20017082201ULL, 20080055111ULL, 20097742781ULL, 20120205221ULL, 20124469481ULL, 20135190191ULL,
	20142157151ULL, 20143868861ULL, 20173988951ULL, 20209214141ULL, 20240175071ULL, 20252937821ULL, 20275400261ULL,
	20297352191ULL, 20343298091ULL, 20366961731ULL, 20390445191ULL, 20393508251ULL, 20464979651ULL, 20471946611ULL,
	20476030691ULL, 20482667321ULL, 20483358011ULL, 20566390961ULL, 20594138681ULL, 20600775311ULL, 20607742271ULL,
	20616601121ULL, 20636000501ULL, 20637862361ULL, 20699123561ULL, 20707471901ULL, 20740985381ULL, 20743537931ULL,
	20778432791ULL, 20812456781ULL, 20823868181ULL, 20830835141ULL, 20839693991ULL, 20843267561ULL, 20859934211ULL,
	20860955231ULL, 20941615811ULL, 20953027211ULL, 20979063221ULL, 21001525661ULL, 21053928011ULL, 21059723801ULL,
	21083027081ULL, 21084048101ULL, 21134258261ULL, 21141225221ULL, 21156720701ULL, 21167771741ULL, 21195519461ULL,
	21224618531ULL, 21237381281ULL, 21277020881ULL, 21282816671ULL, 21300504341ULL, 21344918711ULL, 21356330111ULL,
	21364318091ULL, 21394438181ULL, 21410263991ULL, 21411975701ULL, 21447711401ULL, 21460474151ULL, 21482426081ULL,
	21552696281ULL, 21580444001ULL, 21587410961ULL, 21615669191ULL, 21617531051ULL, 21683567021ULL, 21685428881ULL,
	21696840281ULL, 21703807241ULL, 21753326711ULL, 21822936251ULL, 21837050351ULL, 21906659891ULL, 21919933151ULL,
	21927921131ULL, 21975398561ULL, 21993957101ULL, 22011644771ULL, 22044317411ULL, 22045008101ULL, 22049092181ULL,
	22060143221ULL, 22068491561ULL, 22093506551ULL, 22155788771ULL, 22181824781ULL, 22217049971ULL, 22221974891ULL,
	22256689571ULL, 22315908731ULL };

#ifndef WIN32

#define DWORD uint32_t

__inline DWORD GetTickCount()
{
	struct timeval tv; 
	gettimeofday(&tv, 0); 
	uint64_t res = uint64_t( tv.tv_sec ) * 1000 + tv.tv_usec / 1000;
	return (DWORD)res;
}

#define ULONGLONG uint64_t

__inline ULONGLONG GetTickCount64()

{
	struct timeval tv; 
	gettimeofday(&tv, 0); 
	uint64_t res = uint64_t( tv.tv_sec ) * 1000 + tv.tv_usec / 1000;
	return (ULONGLONG)res;
}

        
#endif


inline int64 GetTimeMicros() 
{ 
	return (boost::posix_time::ptime(boost::posix_time::microsec_clock::universal_time()) - boost::posix_time::ptime(boost::gregorian::date(1970,1,1))).total_microseconds(); 
} 



uint64 sqrtld(uint64 N) {
	int                 b = 1;
	uint64       res,s;
	while(1ULL<<b<N) b+= 1;
	res = 1ULL<<(b/2 + 1);
	for(;;) {
		s = (N/res + res)/2;
		if(s>=res) return res;
		res = s;
	}
}

uint64 mpz2ull(mpz_t z)
{
	uint64 result = 0;
	mpz_export(&result, 0, 0, sizeof(uint64), 0, 0, z);
	return result;
}
 
unsigned int * make_primes(unsigned int limit) {
	std::vector<uint32_t> primevec;
	primesieve::generate_n_primes(limit, &primevec);
	primes = (unsigned int*)malloc((limit + 1) * sizeof(unsigned int));
	primes[0] = limit;
	memcpy(&primes[1], &primevec[0], limit*sizeof(uint32_t));
	return primes;
}

#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )

namespace Core
{
	/** Divisor bit_array_sieve for Prime Searching. **/
	std::vector<unsigned int> DIVISOR_SIEVE;
	void fermat_gpu_benchmark();
	void InitializePrimes()
	{		
		printf("\nGenerating primes...\n");
		primes = make_primes(nPrimeLimit);
		printf("%d primes generated\n", primes[0]);

		mpz_init(zPrimorial);
		mpz_set_ui(zPrimorial, 1);
		double max_sieve = pow(2.0, 64);
		for (unsigned int i=1; i<nPrimorialEndPrime; i++)
		{
			mpz_mul_ui(zPrimorial, zPrimorial, primes[i]);
			max_sieve /= primes[i];
		}
		gmp_printf("\nPrimorial: %Zd\n", zPrimorial);

		printf("Last Primorial Prime = %u\n", primes[nPrimorialEndPrime-1]);
		printf("First Sieving Prime = %u\n", primes[nPrimorialEndPrime]);

		int nSize = (int)mpz_sizeinbase(zPrimorial,2);
		printf("Primorial Size = %d-bit\n", nSize);
		printf("Max. sieve size: %" PRIu64 " bits\n", (uint64_t)max_sieve);

		inverses=(unsigned int *) malloc((nPrimeLimit+1)*sizeof(unsigned int));
		memset(inverses, 0, (nPrimeLimit+1) * sizeof(unsigned int));

		mpz_t zPrime, zInverse, zResult;

		mpz_init(zPrime);
		mpz_init(zInverse);
		mpz_init(zResult);

		printf("\nGenerating inverses...\n");

		for(unsigned int i=nPrimorialEndPrime; i<=nPrimeLimit; i++)
		{
			mpz_set_ui(zPrime, primes[i]);

			int	inv = mpz_invert(zResult, zPrimorial, zPrime);
			if (inv <= 0)
			{
				printf("\nNo Inverse for prime %u at position %u\n\n", primes[i], i);
				exit(0);
			}
			else
			{
				inverses[i]  = (unsigned int)mpz_get_ui(zResult);
			}
		}

		printf("%d inverses generated\n\n", nPrimeLimit - nPrimorialEndPrime + 1);

		printf("\nGenerating invK...\n");
		invK = (uint64_t*)malloc((nPrimeLimit + 1) * sizeof(uint64_t));
		memset(invK, 0, (nPrimeLimit + 1) * sizeof(uint64_t));

		mpz_t n1, n2;
		mpz_init(n1);
		mpz_init(n2);

		mpz_set_ui(n1, 2);
		mpz_pow_ui(n1, n1, 64);

		for (unsigned int i = nPrimorialEndPrime; i <= nPrimeLimit; i++)
		{
			mpz_div_ui(n2, n1, primes[i]);
			uint64_t recip = mpz2ull(n2);			
			invK[i] = recip;
		}

		mpz_clear(n1);
		mpz_clear(n2);
	}
	
	/** Convert Double to unsigned int Representative. Used for encoding / decoding prime difficulty from nBits. **/
	unsigned int SetBits(double nDiff)
	{
		unsigned int nBits = 10000000;
		nBits = (unsigned int)(nBits * nDiff);
		
		return nBits;
	}

	/** Determines the difficulty of the Given Prime Number.
		Difficulty is represented as so V.X
		V is the whole number, or Cluster Size, X is a proportion
		of Fermat Remainder from last Composite Number [0 - 1] **/
	double GetPrimeDifficulty(CBigNum prime, int checks)
	{
		CBigNum lastPrime = prime;
		CBigNum next = prime + 2;
		unsigned int clusterSize = 1;
		
		///largest prime gap in cluster can be +12
		///this was determined by previously found clusters up to 17 primes
		for( next ; next <= lastPrime + 12; next += 2)
		{
			if(PrimeCheck(next, checks))
			{
				lastPrime = next;
				++clusterSize;
			}
		}
		
		///calulate the rarety of cluster from proportion of fermat remainder of last prime + 2
		///keep fractional remainder in bounds of [0, 1]
		double fractionalRemainder = 1000000.0 / GetFractionalDifficulty(next);
		if(fractionalRemainder > 1.0 || fractionalRemainder < 0.0)
			fractionalRemainder = 0.0;
		
		return (clusterSize + fractionalRemainder);
	}

	double GetPrimeDifficulty2(CBigNum next, unsigned int clusterSize)
	{
		///calulate the rarety of cluster from proportion of fermat remainder of last prime + 2
		///keep fractional remainder in bounds of [0, 1]
		double fractionalRemainder = 1000000.0 / GetFractionalDifficulty(next);
		if(fractionalRemainder > 1.0 || fractionalRemainder < 0.0)
			fractionalRemainder = 0.0;
		
		return (clusterSize + fractionalRemainder);
	}

	/** Gets the unsigned int representative of a decimal prime difficulty **/
	unsigned int GetPrimeBits(CBigNum prime, int checks)
	{
		return SetBits(GetPrimeDifficulty(prime, checks));
	}

	/** Breaks the remainder of last composite in Prime Cluster into an integer. 
		Larger numbers are more rare to find, so a proportion can be determined 
		to give decimal difficulty between whole number increases. **/
	unsigned int GetFractionalDifficulty(CBigNum composite)
	{
		/** Break the remainder of Fermat test to calculate fractional difficulty [Thanks Sunny] **/
		return ((composite - FermatTest(composite, 2) << 24) / composite).getuint();
	}
	
	
	/** bit_array_sieve of Eratosthenes for Divisor Tests. Used for Searching Primes. **/
	std::vector<unsigned int> Eratosthenes(int nSieveSize)
	{
		bool *TABLE = new bool[nSieveSize];
		
		for(int nIndex = 0; nIndex < nSieveSize; nIndex++)
			TABLE[nIndex] = false;
			
			
		for(int nIndex = 2; nIndex < nSieveSize; nIndex++)
			for(int nComposite = 2; (nComposite * nIndex) < nSieveSize; nComposite++)
				TABLE[nComposite * nIndex] = true;
		
		
		std::vector<unsigned int> PRIMES;
		for(int nIndex = 2; nIndex < nSieveSize; nIndex++)
			if(!TABLE[nIndex])
				PRIMES.push_back(nIndex);

		
		printf("bit_array_sieve of Eratosthenes Generated %u Primes.\n", (unsigned int)PRIMES.size());
		
		delete[] TABLE;
		return PRIMES;
	}
	
	/** Basic Search filter to determine if further tests should be done. **/
	bool DivisorCheck(CBigNum test)
	{
		for(int index = 0; index < DIVISOR_SIEVE.size(); index++)
			if(test % DIVISOR_SIEVE[index] == 0)
				return false;
				
		return true;
	}

	/** Determines if given number is Prime. Accuracy can be determined by "checks". 
		The default checks the Nexus Network uses is 2 **/
	bool PrimeCheck(CBigNum test, int checks)
	{
		/** Check C: Fermat Tests */
		CBigNum n = 3;
		if(FermatTest(test, n) != 1)
				return false;
		
		return true;
	}

	/** Simple Modular Exponential Equation a^(n - 1) % n == 1 or notated in Modular Arithmetic a^(n - 1) = 1 [mod n]. 
		a = Base or 2... 2 + checks, n is the Prime Test. Used after Miller-Rabin and Divisor tests to verify primality. **/
	CBigNum FermatTest(CBigNum n, CBigNum a)
	{
		CAutoBN_CTX pctx;
		CBigNum e = n - 1;
		CBigNum r;
		BN_mod_exp(&r, &a, &e, &n, pctx);
		
		return r;
	}

	/** Miller-Rabin Primality Test from the OpenSSL BN Library. **/
	bool Miller_Rabin(CBigNum n, int checks)
	{
		return (BN_is_prime(&n, checks, NULL, NULL, NULL) == 1);
	}

	unsigned int mpi_mod_int(mpz_t A, unsigned int B)
	{
		if (B == 1)
			return 0;
		else if (B == 2)
			return A[0]._mp_d[0]&1;

		#define biH (sizeof(mp_limb_t)<<2)
		int i;
		mp_limb_t b=B,x,y,z;

		for( i = A[0]._mp_alloc - 1, y = 0; i > 0; i-- )
		{
			x  = A[0]._mp_d[i - 1];
			y  = ( y << biH ) | ( x >> biH );
			z  = y / b;
			y -= z * b;

			x <<= biH;
			y  = ( y << biH ) | ( x >> biH );
			z  = y / b;
			y -= z * b;
		}

		return (unsigned int)y;
	}

	static int Convert_BIGNUM_to_mpz_t(const BIGNUM *bn, mpz_t g)
	{
		bn_check_top(bn);
		if(((sizeof(bn->d[0]) * 8) == GMP_NUMB_BITS) &&
				(BN_BITS2 == GMP_NUMB_BITS)) 
		{
			/* The common case */
			if(!_mpz_realloc (g, bn->top))
				return 0;
			memcpy(&g->_mp_d[0], &bn->d[0], bn->top * sizeof(bn->d[0]));
			g->_mp_size = bn->top;
			if(bn->neg)
				g->_mp_size = -g->_mp_size;
			return 1;
		}
		else
		{
			char *tmpchar = BN_bn2hex(bn);
			if(!tmpchar) return 0;
			OPENSSL_free(tmpchar);
			return 0;
		}
	}

	boost::mutex work_mutex;
	std::deque<work_info> work_queue;
	std::queue<work_info> result_queue;	

	int scan_offsets(work_info &work)
	{
		int scanned = 0;

		work.nNonce = false;
		work.nNonceDifficulty = 0;

		mpz_t zTempVar, zN, zFirstSieveElement, zPrimeOrigin, zPrimeOriginOffset, zResidue, zTwo;
		mpz_init(zTempVar);
		mpz_init(zN);
		mpz_init_set(zFirstSieveElement, work.zFirstSieveElement.__get_mp());
		mpz_init(zPrimeOrigin);
		Convert_BIGNUM_to_mpz_t(&work.BaseHash, zPrimeOrigin);
		mpz_init(zPrimeOriginOffset);
		mpz_init(zResidue);
		mpz_init_set_ui(zTwo, 2);

		mpz_mod(zTempVar, zFirstSieveElement, zPrimorial);
		uint64_t constellation_origin = mpz_get_ui(zTempVar);

		uint64_t nNonce = 0;
		unsigned int nPrimeCount = 0;
		unsigned int nSieveDifficulty = 0;
		uint64_t nStart = 0;
		uint64_t nStop = 0;
		unsigned int nLastOffset = 0;

		std::vector<int> offsets;
		offsets.reserve(8);
		for(unsigned int i=0; i<work.nonce_offsets.size();i++)
		{
			if(work.nHeight != nBestHeight)
			{
				break;
			}

			uint32_t offset = work.nonce_offsets[i];
			if (offset != 0xFFFFFFFF)
			{
				scanned++;
				mpz_mul_ui(zTempVar, zPrimorial, offset);
				mpz_add(zTempVar, zFirstSieveElement, zTempVar);
				mpz_set(zPrimeOriginOffset, zTempVar);

				nStart = 0;
				nStop = 0;
				nPrimeCount = 0;
				nLastOffset = 0;

				uint64_t mask = 0;
				int masklen = 0;
				bool bad = false;
				offsets.clear();
				for(; nStart<=nStop+12; nStart+=2, mpz_add_ui(zTempVar, zTempVar, 2), masklen++)
				{
					bool prime = false;
					bool possible_prime = true;
					if (masklen == 2 && mask == 0b11) possible_prime = false;
					else if (masklen == 4 && (mask == 0b1010 || mask == 0b1110 )) possible_prime = false;
					mask <<= 1;

					if ((constellation_origin+2*masklen) % 3 == 0) possible_prime = false;
					else if ((constellation_origin+2*masklen) % 5 == 0) possible_prime = false;
					else if ((constellation_origin+2*masklen) % 7 == 0) possible_prime = false;

					if (possible_prime)
					{
						/* Miller-Rabin */
						/*
						bool Miller_Rabin(CBigNum n, int checks)
						{
							return (BN_is_prime(&n, checks, NULL, NULL, NULL) == 1);
						}
	*/

						if (nPrimeCount == 0) PrimesChecked++;

						mpz_sub_ui(zN, zTempVar, 1);
						mpz_powm(zResidue, zTwo, zN, zTempVar);
						if (mpz_cmp_ui(zResidue, 1) == 0)
						{
							if (nPrimeCount == 0) PrimesFound++;
							nStop = nStart;
							nPrimeCount++;
							offsets.push_back(nStart);
							mask |= 1;
							prime = true;
						}
					}

					if (!prime && nStart == 0)
					{
						bad = true;
						break;
					}
					nLastOffset += 2;
				}

				if (bad) {
					continue;
				}

				nSieveDifficulty = 0;
				if (nPrimeCount >= 4)
				{
					mpz_sub(zTempVar, zPrimeOriginOffset, zPrimeOrigin);
					nNonce = mpz_get_ui(zTempVar);
					nSieveDifficulty = SetBits(GetPrimeDifficulty2(work.BaseHash + nNonce + nLastOffset, nPrimeCount));
				}

				if (nSieveDifficulty >= 80000000)
					nEightChainsFoundCounter++;
				else if (nSieveDifficulty >= 70000000)
					nSevenChainsFoundCounter++;
				else if (nSieveDifficulty >= 60000000)
					nSixChainsFoundCounter++;
				else if (nSieveDifficulty >= 50000000)
					nFiveChainsFoundCounter++;
				else if (nSieveDifficulty >= 40000000)
					nFourChainsFoundCounter++;

				if(nSieveDifficulty >= 40000000) {
					if (nSieveDifficulty >= 60000000)
						printf("\n  %d-Chain found: diff %f - origin: %" PRIu64 "\n", (int)offsets.size(), (double)nSieveDifficulty / 1e7, constellation_origin);
				}

				if(nSieveDifficulty >= work.nDifficulty)
				{
					work.nNonce = nNonce;
					work.nNonceDifficulty = nSieveDifficulty;
					break;
				}
			} else PrimesChecked++;
		}
		SievedBits += nBitArray_Size[work.gpu_thread];
		CheckedCandidates += work.nonce_offsets.size();
#if TIMING
		QueryPerformanceCounter(&work.EndingTime);
#endif

		mpz_clear(zPrimeOrigin);
		mpz_clear(zPrimeOriginOffset);
		mpz_clear(zFirstSieveElement);
		mpz_clear(zResidue);
		mpz_clear(zTwo);
		mpz_clear(zN);
		mpz_clear(zTempVar);

		return scanned;
	}

	bool PrimeQuery()
	{
		work_info work;
		bool have_work = false;
		{
			boost::mutex::scoped_lock lock(work_mutex);
			if (!work_queue.empty())
			{
				work = work_queue.front();
				work_queue.pop_front();
				have_work = true;
			}
		}

		if (have_work)
		{
			int scanned = scan_offsets(work);

			if (work.nNonce != 0 && work.nNonceDifficulty > work.nDifficulty)
			{
				boost::mutex::scoped_lock lock(work_mutex);
				result_queue.emplace(work);
			}
		}
		return have_work;
	}

	void Adjust_Sieve(int gpu_thread, size_t queue_size)
	{		
		int target = 4*nThreadsCPU;
		unsigned int tmp;
		double factor = 1.0 + ((double)((int)queue_size - target)/(target*5));
		if (factor > 1) factor = factor*factor;
		else factor = sqrt(factor);
		tmp = (unsigned int)((double)nPrimeLimitB[gpu_thread]*factor);
		if (tmp < 10000) tmp = 10000;
		else if (tmp > nPrimeLimit) tmp = nPrimeLimit;
		nPrimeLimitB[gpu_thread] = tmp;
	}

	void PrimeSieve(int threadIndex, CBigNum BaseHash, unsigned int nDifficulty, unsigned int nHeight, uint512 merkeRoot)
	{
		uint64_t result = false;

		if (!cuda_init(threadIndex))
		{
			Sleep(1000 * threadIndex);
			fprintf(stderr, "Thread %d starting up...\n", threadIndex);

			cuda_set_primes(threadIndex, primes, inverses, invK, nPrimeLimit, nBitArray_Size[threadIndex], 1024);			
			compaction_gpu_init(threadIndex, nBitArray_Size[threadIndex]);
		}

		mpz_t zPrimeOrigin, zFirstSieveElement, zPrimorialMod, zTempVar;

		unsigned int i = 0;
		unsigned int j = 0;
		unsigned int nSize = 0;

		mpz_init(zFirstSieveElement);
		mpz_init(zPrimorialMod);

		mpz_init(zTempVar);
		mpz_init(zPrimeOrigin);

		Convert_BIGNUM_to_mpz_t(&BaseHash, zPrimeOrigin);
		nSize = mpz_sizeinbase(zPrimeOrigin,2);

		static unsigned char* static_bit_array_sieve[8] = {0,0,0,0,0,0,0,0};
		if (static_bit_array_sieve[threadIndex] == NULL)
			static_bit_array_sieve[threadIndex] = (unsigned char*)malloc((nBitArray_Size[threadIndex])/8);

		static uint32_t* static_nonce_offsets[8] = {0,0,0,0,0,0,0,0};
		if (static_nonce_offsets[threadIndex] == NULL)
			static_nonce_offsets[threadIndex] = (uint32_t*)malloc((nBitArray_Size[threadIndex]>>3) * sizeof(uint32_t));

		unsigned char* bit_array_sieve = static_bit_array_sieve[threadIndex];
		unsigned int *base_remainders = new unsigned int[nPrimeLimit];

		mpz_mod(zPrimorialMod, zPrimeOrigin, zPrimorial);
		mpz_sub(zPrimorialMod, zPrimorial, zPrimorialMod);
		mpz_mod(zPrimorialMod, zPrimorialMod, zPrimorial);
		mpz_add(zTempVar, zPrimeOrigin, zPrimorialMod);

		cuda_set_zTempVar(threadIndex, (const uint64_t*)zTempVar[0]._mp_d);
		cuda_compute_base_remainders(threadIndex, base_remainders, nPrimorialEndPrime, nPrimeLimit);

		cuda_set_origins(threadIndex, &originTable[0], 1024, nSharedSizeKB[threadIndex], nPrimorialEndPrime, nPrimeLimitA[threadIndex]);
		int originCounter = 0;
		for (auto search: originTable)
		{
			const uint64_t origin = search;

			uint64_t base_offset = origin;
			unsigned int primeLimit = nPrimeLimitB[threadIndex];
			
			memset(bit_array_sieve, 0x00, (nBitArray_Size[threadIndex])/8);

			mpz_mod(zPrimorialMod, zPrimeOrigin, zPrimorial);
			mpz_sub(zPrimorialMod, zPrimorial, zPrimorialMod);

			mpz_mod(zPrimorialMod, zPrimorialMod, zPrimorial);
			mpz_add_ui(zPrimorialMod, zPrimorialMod, origin);
			mpz_add(zTempVar, zPrimeOrigin, zPrimorialMod);

			mpz_set(zFirstSieveElement, zTempVar);
			cuda_set_zFirstSieveElement(threadIndex, (const uint64_t*)zFirstSieveElement[0]._mp_d);

			cuda_compute_primesieve(threadIndex, nSharedSizeKB[threadIndex], nThreadsKernelA[threadIndex], 
				bit_array_sieve, base_remainders, base_offset, originCounter, nPrimorialEndPrime, nPrimeLimitA[threadIndex], nPrimeLimitB[threadIndex],
				nBitArray_Size[threadIndex], nDifficulty, 0);

			size_t numberOfCandidates = 0;
			compaction_gpu(threadIndex, nBitArray_Size[threadIndex], static_nonce_offsets[threadIndex], &numberOfCandidates);

			unsigned int max_queue = 4*nThreadsCPU;
			if(work_queue.size() >= max_queue)
			{
				fprintf(stderr, ".");

				cuda_compute_fermat(threadIndex, static_nonce_offsets[threadIndex], numberOfCandidates, 0, mpz_get_ui(zPrimorial));
				int count = 0;
				for (int i=0; i < numberOfCandidates; i++)
					if (static_nonce_offsets[threadIndex][i] != 0xFFFFFFFF) count++;

			}

			if(nHeight != nBestHeight)
			{
				goto request_new_block;
			}

			{
				boost::mutex::scoped_lock lock(work_mutex);
				work_queue.emplace_back(work_info(BaseHash, nDifficulty, nHeight, threadIndex, static_nonce_offsets[threadIndex], static_nonce_offsets[threadIndex] + numberOfCandidates, zFirstSieveElement, merkeRoot, primeLimit));
				
			}

			originCounter++;
		}

request_new_block:

		mpz_clear(zPrimeOrigin);
		mpz_clear(zFirstSieveElement);
		mpz_clear(zPrimorialMod);
		mpz_clear(zTempVar);

		delete[] base_remainders;
	}
}

