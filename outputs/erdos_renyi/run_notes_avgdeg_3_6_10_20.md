# ER Phase C Run Log

Date: 2026-07-07

## Parameter Snapshot

```python
N = 5000
ACTIVE_AVERAGE_DEGREES = [3, 6, 10, 20]
ALL_AVERAGE_DEGREES = [3, 6, 10, 20]
NUM_DENSITY_GRAPHS = 30
NUM_SPACING_GRAPHS = 30
DENSITY_SEED = 31415
SPACING_SEED = 27182
COMPONENT_MODE = "largest_connected_component"
DENSITY_METHOD = "local_count"
LOCAL_COUNT_HALF_WIDTH = 0.15
DENSITY_NUM_POINTS = 2001
DENSITY_NUM_BINS = 120
DENSITY_BANDWIDTH_SCALE = 1.0
EXPLICIT_E_VALUES = [-2.5, -1.5, 0.0, 1.5, 2.5]
QUANTILES = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]
WINDOW_HALF_WIDTH = 0.15
NUM_BINS = 30
MIN_EIGENVALUES = 10
MIN_TOTAL_GAPS = 100
SMALL_SPACING_THRESHOLD = 0.20
KS_SAMPLE_SIZE = 4096
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "erdos_renyi"
```

## Command

```bash
python3 src/small_world_mobility_edge/erdos_renyi_spacing_by_energy.py
```

## Generated Output Directories

- [avgdeg_3](../../outputs/erdos_renyi/avgdeg_3)
- [avgdeg_6](../../outputs/erdos_renyi/avgdeg_6)
- [avgdeg_10](../../outputs/erdos_renyi/avgdeg_10)
- [avgdeg_20](../../outputs/erdos_renyi/avgdeg_20)

Per degree, the saved files are:

- `global_empirical_density.png`
- `local_spacing_grid_explicit_E.png`
- `local_spacing_grid_quantiles.png`
- `local_spacing_summary_explicit_E.png`
- `local_spacing_summary_quantiles.png`
- `spacing_summary.csv`

## Full Numerical Output

The rows below preserve the current `spacing_summary.csv` contents for each average degree.

### avgdeg = 3

| average_degree | E | spectral_quantile | window_min | window_max | number_of_graphs | graphs_used | graphs_skipped | number_of_eigenvalues | number_of_gaps | rho_hat_E | mean_spacing | mean_gap_ratio | small_spacing_probability | ks_distance_to_goe | ks_distance_to_poisson | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 3 | -2.5 | 0.09237331157216479 | -2.65 | -2.35 | 30 | 30 | 0 | 4656 | 4626 | 0.1105471561112534 | 1.0038582022558047 | 0.5302718675934167 | 0.03415477734543882 | 0.01837440553393861 | 0.21357544314742757 | ok |
| 3 | -1.5 | 0.22802441415392152 | -1.65 | -1.35 | 30 | 30 | 0 | 6698 | 6668 | 0.15849363700952945 | 0.9996572876121759 | 0.5286594094607875 | 0.03269346130773845 | 0.01229754049190162 | 0.20395920815836832 | ok |
| 3 | 0.0 | 0.5001786656056793 | -0.15 | 0.15 | 30 | 30 | 0 | 11676 | 11646 | 0.2733953215770685 | 0.9885307381519775 | 0.48399652554287803 | 0.30139103554868624 | 0.28499055469689166 | 0.284732955521209 | ok |
| 3 | 1.5 | 0.7722387700997516 | 1.35 | 1.65 | 30 | 30 | 0 | 6700 | 6670 | 0.1586385870929439 | 1.0017385146836806 | 0.5292151712893817 | 0.03163418290854573 | 0.019790104947526237 | 0.21154422788605698 | ok |
| 3 | 2.5 | 0.9078673023550974 | 2.35 | 2.65 | 30 | 30 | 0 | 4662 | 4632 | 0.11057062944593174 | 1.002007346198959 | 0.5373500137943417 | 0.033246977547495685 | 0.025690846286701208 | 0.21135578583765113 | ok |
| 3 | -2.933158869246636 | 0.05 | -3.083158869246636 | -2.7831588692466362 | 30 | 30 | 0 | 3582 | 3552 | 0.08437557988648736 | 0.9952349854140462 | 0.5288218826150498 | 0.03209459459459459 | 0.020424232826576575 | 0.20501873944256757 | ok |
| 3 | -2.432187498735415 | 0.1 | -2.582187498735415 | -2.282187498735415 | 30 | 30 | 0 | 4823 | 4793 | 0.11425960753747864 | 1.001898683012561 | 0.527057885976091 | 0.03192155226371792 | 0.019820571667014397 | 0.21781765074066348 | ok |
| 3 | -1.6811060271048985 | 0.2 | -1.8311060271048984 | -1.5311060271048986 | 30 | 30 | 0 | 6383 | 6353 | 0.15081357032982484 | 0.9988726715827119 | 0.5308072254353026 | 0.031166378089091767 | 0.024555328191405636 | 0.21092397292617662 | ok |
| 3 | -0.7781072803889741 | 0.35 | -0.9281072803889742 | -0.6281072803889741 | 30 | 30 | 0 | 7392 | 7362 | 0.17482242149261104 | 0.9992417553208506 | 0.52900497293387 | 0.029883183917413746 | 0.02336321651725075 | 0.20809562618853572 | ok |
| 3 | -0.0006534831400295194 | 0.5 | -0.15065348314002952 | 0.14934651685997047 | 30 | 30 | 0 | 11679 | 11649 | 0.2734285465236042 | 0.9889920933626596 | 0.4839892101080216 | 0.30131341746072626 | 0.2846596274358314 | 0.2846596274358314 | ok |
| 3 | 0.7763966505838037 | 0.65 | 0.6263966505838037 | 0.9263966505838037 | 30 | 30 | 0 | 7410 | 7380 | 0.17487669088276064 | 0.9977156874401824 | 0.5322688357548137 | 0.030758807588075882 | 0.020596205962059622 | 0.2078590785907859 | ok |
| 3 | 1.6793882176631287 | 0.8 | 1.5293882176631288 | 1.8293882176631286 | 30 | 30 | 0 | 6387 | 6357 | 0.15091953508322992 | 0.9987802745413192 | 0.5313870971073978 | 0.03287714330659116 | 0.02343872896020135 | 0.21881390593047034 | ok |
| 3 | 2.4301122551139835 | 0.9 | 2.2801122551139836 | 2.5801122551139835 | 30 | 30 | 0 | 4816 | 4786 | 0.11435842636264838 | 1.0027479842361182 | 0.53890323266125 | 0.028416213957375678 | 0.01567070622649394 | 0.22294191391558713 | ok |
| 3 | 2.930901637796532 | 0.95 | 2.780901637796532 | 3.080901637796532 | 30 | 30 | 0 | 3577 | 3547 | 0.0842785532841302 | 0.994842041971641 | 0.5318512276016709 | 0.03354947843247815 | 0.02929673733965323 | 0.20201889911368762 | ok |

### avgdeg = 6

| average_degree | E | spectral_quantile | window_min | window_max | number_of_graphs | graphs_used | graphs_skipped | number_of_eigenvalues | number_of_gaps | rho_hat_E | mean_spacing | mean_gap_ratio | small_spacing_probability | ks_distance_to_goe | ks_distance_to_poisson | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 6 | -2.5 | 0.18148796204502504 | -2.65 | -2.35 | 30 | 30 | 0 | 4695 | 4665 | 0.10482338832393402 | 1.00046261033834 | 0.5222370040599176 | 0.036655948553054665 | 0.01564844587352626 | 0.21564844587352627 | ok |
| 6 | -1.5 | 0.2957307345576469 | -1.65 | -1.35 | 30 | 30 | 0 | 5518 | 5488 | 0.12340597641025206 | 1.0014355980558445 | 0.533656898396833 | 0.026785714285714284 | 0.01639941690962099 | 0.22284985422740525 | ok |
| 6 | 0.0 | 0.5001508041772084 | -0.15 | 0.15 | 30 | 30 | 0 | 6950 | 6920 | 0.15521192456659816 | 1.001572419057004 | 0.5272284782729851 | 0.03265895953757225 | 0.014884393063583815 | 0.20895953757225433 | ok |
| 6 | 1.5 | 0.704625989223492 | 1.35 | 1.65 | 30 | 30 | 0 | 5530 | 5500 | 0.12338535140945202 | 0.9998003837277458 | 0.5332722730950028 | 0.032545454545454544 | 0.02090909090909091 | 0.21454545454545454 | ok |
| 6 | 2.5 | 0.818878588973468 | 2.35 | 2.65 | 30 | 30 | 0 | 4687 | 4657 | 0.10469368894120754 | 1.000962385200029 | 0.532202769191527 | 0.03242430749409491 | 0.01932574618853339 | 0.2044234485720421 | ok |
| 6 | -4.024715667099305 | 0.05 | -4.1747156670993055 | -3.8747156670993053 | 30 | 30 | 0 | 2900 | 2870 | 0.06471581132322207 | 0.9972550403354383 | 0.5341612688888434 | 0.03484320557491289 | 0.025949000979965157 | 0.23032481761759582 | ok |
| 6 | -3.357950130103271 | 0.1 | -3.507950130103271 | -3.207950130103271 | 30 | 30 | 0 | 3783 | 3753 | 0.0845482502075214 | 0.9991615297253689 | 0.5333432797920876 | 0.03543831601385558 | 0.0205871110403344 | 0.2101478322550293 | ok |
| 6 | -2.3260793839084846 | 0.2 | -2.4760793839084845 | -2.1760793839084847 | 30 | 30 | 0 | 4845 | 4815 | 0.10822420610848649 | 1.001205718089188 | 0.5292087795916206 | 0.03717549325025961 | 0.01806853582554517 | 0.21329179646936655 | ok |
| 6 | -1.0714096771980455 | 0.35 | -1.2214096771980454 | -0.9214096771980455 | 30 | 30 | 0 | 5846 | 5816 | 0.13008442114457774 | 0.9965498713029902 | 0.5303859722851133 | 0.029573590096286108 | 0.026822558459422285 | 0.2123452544704264 | ok |
| 6 | -0.0009716483415724065 | 0.5 | -0.1509716483415724 | 0.1490283516584276 | 30 | 30 | 0 | 6945 | 6915 | 0.15520840222757595 | 1.001722276870096 | 0.5268857386971159 | 0.0326825741142444 | 0.02010122921185828 | 0.22313810556760666 | ok |
| 6 | 1.0692258944530533 | 0.65 | 0.9192258944530532 | 1.2192258944530532 | 30 | 30 | 0 | 5842 | 5812 | 0.1301435594368497 | 0.9978900285898102 | 0.5328700074910105 | 0.03320715760495527 | 0.012904335856847901 | 0.21954576737783896 | ok |
| 6 | 2.3226539722149475 | 0.8 | 2.1726539722149476 | 2.4726539722149474 | 30 | 30 | 0 | 4860 | 4830 | 0.1084623606164855 | 1.0000916586255626 | 0.5335085017142773 | 0.033747412008281574 | 0.02443064182194617 | 0.22132505175983436 | ok |
| 6 | 3.354892270057118 | 0.9 | 3.204892270057118 | 3.504892270057118 | 30 | 30 | 0 | 3785 | 3755 | 0.08432909119531438 | 0.9981030280871491 | 0.5336200866449676 | 0.02822902796271638 | 0.025152856087716377 | 0.2306586010319574 | ok |
| 6 | 4.0213679413882195 | 0.95 | 3.8713679413882196 | 4.17136794138822 | 30 | 30 | 0 | 2904 | 2874 | 0.06476305431896785 | 0.995661215217122 | 0.529213552333583 | 0.03409881697981907 | 0.022373950721990256 | 0.21694366518789143 | ok |

### avgdeg = 10

| average_degree | E | spectral_quantile | window_min | window_max | number_of_graphs | graphs_used | graphs_skipped | number_of_eigenvalues | number_of_gaps | rho_hat_E | mean_spacing | mean_gap_ratio | small_spacing_probability | ks_distance_to_goe | ks_distance_to_poisson | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 10 | -2.5 | 0.24772022076516537 | -2.65 | -2.35 | 30 | 30 | 0 | 4091 | 4061 | 0.09109696412323377 | 0.9978262569472087 | 0.53245823142724 | 0.03225806451612903 | 0.020688468145930804 | 0.20536687344173232 | ok |
| 10 | -1.5 | 0.3433022090510972 | -1.65 | -1.35 | 30 | 30 | 0 | 4484 | 4454 | 0.0998010244029335 | 0.9995417095185979 | 0.5268685004435186 | 0.03435114503816794 | 0.022227211495285136 | 0.1984732824427481 | ok |
| 10 | 0.0 | 0.500201023387484 | -0.15 | 0.15 | 30 | 30 | 0 | 4885 | 4855 | 0.10824229018986249 | 0.9957019727536623 | 0.5281021150686476 | 0.030895983522142123 | 0.012564366632337795 | 0.21462409886714728 | ok |
| 10 | 1.5 | 0.6571302759220252 | 1.35 | 1.65 | 30 | 30 | 0 | 4485 | 4455 | 0.0996854848537208 | 0.9982260138616879 | 0.5376562445272994 | 0.029405162738496072 | 0.013692480359147026 | 0.2101010101010101 | ok |
| 10 | 2.5 | 0.7526384362032024 | 2.35 | 2.65 | 30 | 30 | 0 | 4091 | 4061 | 0.09060946314317886 | 0.9938085300125998 | 0.5237833114407613 | 0.03718295986210293 | 0.020433686572734547 | 0.2073430838886666 | ok |
| 10 | -5.159341537995145 | 0.05 | -5.309341537995145 | -5.0093415379951445 | 30 | 30 | 0 | 2389 | 2359 | 0.053122469368363824 | 0.9959148631099806 | 0.5372887470733803 | 0.040695209834675714 | 0.01674800547769182 | 0.22290028713172955 | ok |
| 10 | -4.337679352411802 | 0.1 | -4.487679352411802 | -4.187679352411801 | 30 | 30 | 0 | 3055 | 3025 | 0.06781686339945601 | 0.9952893848228072 | 0.5396006739503756 | 0.03239669421487603 | 0.017371916967975208 | 0.21628413933367768 | ok |
| 10 | -3.040902432430885 | 0.2 | -3.190902432430885 | -2.890902432430885 | 30 | 30 | 0 | 3845 | 3815 | 0.08531610718977832 | 0.9978233222408229 | 0.5311052636006217 | 0.032241153342070776 | 0.021032634850098295 | 0.20148903383027522 | ok |
| 10 | -1.433030487554086 | 0.35 | -1.583030487554086 | -1.283030487554086 | 30 | 30 | 0 | 4499 | 4469 | 0.100343077263928 | 1.0011488935652737 | 0.5270323417624632 | 0.03669724770642202 | 0.015439695681360483 | 0.20541508167375253 | ok |
| 10 | -0.0018566520089801249 | 0.5 | -0.15185665200898013 | 0.14814334799101986 | 30 | 30 | 0 | 4882 | 4852 | 0.10823094414015866 | 0.9958101177298624 | 0.5283258925285815 | 0.03112118713932399 | 0.029472382522671065 | 0.20507007419620774 | ok |
| 10 | 1.4286689944536426 | 0.65 | 1.2786689944536427 | 1.5786689944536425 | 30 | 30 | 0 | 4495 | 4465 | 0.10034519743024077 | 1.0014592869812082 | 0.537776864745089 | 0.02956326987681971 | 0.013885778275475923 | 0.2461366181410974 | ok |
| 10 | 3.0367947712938173 | 0.8 | 2.8867947712938173 | 3.186794771293817 | 30 | 30 | 0 | 3840 | 3810 | 0.08548978564442243 | 0.9995268773891243 | 0.5335822267716341 | 0.028608923884514435 | 0.022322270751312337 | 0.21358395874343833 | ok |
| 10 | 4.333032516363715 | 0.9 | 4.183032516363714 | 4.483032516363715 | 30 | 30 | 0 | 3062 | 3032 | 0.068102114831756 | 0.9970465658129383 | 0.5319928851390752 | 0.03100263852242744 | 0.021048271561675463 | 0.2113768242908971 | ok |
| 10 | 5.15382510996919 | 0.95 | 5.003825109969189 | 5.30382510996919 | 30 | 30 | 0 | 2400 | 2370 | 0.05317646715090878 | 0.9937419104916129 | 0.5329182103270909 | 0.03713080168776371 | 0.019741602386603375 | 0.20061622329905063 | ok |

### avgdeg = 20

| average_degree | E | spectral_quantile | window_min | window_max | number_of_graphs | graphs_used | graphs_skipped | number_of_eigenvalues | number_of_gaps | rho_hat_E | mean_spacing | mean_gap_ratio | small_spacing_probability | ks_distance_to_goe | ks_distance_to_poisson | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 20 | -2.5 | 0.32061370579459736 | -2.65 | -2.35 | 30 | 30 | 0 | 3106 | 3076 | 0.06899944129373674 | 0.9958538282834368 | 0.5277570035542584 | 0.0377113133940182 | 0.02239902927909623 | 0.2082767164133615 | ok |
| 20 | -1.5 | 0.3910366715768358 | -1.65 | -1.35 | 30 | 30 | 0 | 3230 | 3200 | 0.0716716808640896 | 0.9973693618851249 | 0.5375219231262238 | 0.0328125 | 0.02947265625 | 0.21640625 | ok |
| 20 | 0.0 | 0.5003040864601119 | -0.15 | 0.15 | 30 | 30 | 0 | 3299 | 3269 | 0.07328917368975404 | 0.9965190547610163 | 0.5281891367538256 | 0.032119914346895075 | 0.01392587385763995 | 0.19953346660867238 | ok |
| 20 | 1.5 | 0.609538663299845 | 1.35 | 1.65 | 30 | 30 | 0 | 3230 | 3200 | 0.07168387529899904 | 0.997484879436887 | 0.5302275466913507 | 0.030625 | 0.0215234375 | 0.20189453125 | ok |
| 20 | 2.5 | 0.6799655523680211 | 2.35 | 2.65 | 30 | 30 | 0 | 3103 | 3073 | 0.06880801620535042 | 0.9949376538138829 | 0.5366469291530095 | 0.031239830784249918 | 0.020734236444435405 | 0.20504038767592744 | ok |
| 20 | -7.240936427533932 | 0.05 | -7.390936427533933 | -7.090936427533932 | 30 | 30 | 0 | 1784 | 1754 | 0.03975150360057751 | 0.9981466884191463 | 0.5174226373460636 | 0.03819840364880274 | 0.029222825505986316 | 0.211701378990878 | ok |
| 20 | -6.129520832317734 | 0.1 | -6.2795208323177345 | -5.979520832317734 | 30 | 30 | 0 | 2231 | 2201 | 0.04993895948707009 | 1.0041102669199218 | 0.5187572249828847 | 0.033621081326669695 | 0.021812040336778738 | 0.20582929279304862 | ok |
| 20 | -4.341350720242241 | 0.2 | -4.491350720242242 | -4.191350720242241 | 30 | 30 | 0 | 2762 | 2732 | 0.06135414204172215 | 0.9976568078346671 | 0.5323690844280813 | 0.03184480234260615 | 0.017499842720534408 | 0.22188164863195461 | ok |
| 20 | -2.077694917046921 | 0.35 | -2.227694917046921 | -1.9276949170469213 | 30 | 30 | 0 | 3150 | 3120 | 0.07038426719286847 | 0.9998562490739928 | 0.5306002996964123 | 0.035897435897435895 | 0.016492638221153846 | 0.2081192407852564 | ok |
| 20 | -0.004150303282250826 | 0.5 | -0.15415030328225082 | 0.14584969671774917 | 30 | 30 | 0 | 3297 | 3267 | 0.07326182147772178 | 0.995476942300663 | 0.5281138745079131 | 0.0327517600244873 | 0.022158732208448117 | 0.21025791174051117 | ok |
| 20 | 2.069592585907806 | 0.65 | 1.919592585907806 | 2.219592585907806 | 30 | 30 | 0 | 3171 | 3141 | 0.07021458120427067 | 0.9922718547860518 | 0.5331926485623688 | 0.03183699458771092 | 0.020725681386302132 | 0.21442480126751035 | ok |
| 20 | 4.332413422518878 | 0.8 | 4.182413422518878 | 4.482413422518879 | 30 | 30 | 0 | 2753 | 2723 | 0.06129834726144751 | 0.9961201414759691 | 0.5341507113872463 | 0.030848329048843187 | 0.01119245346355123 | 0.21235859030710613 | ok |
| 20 | 6.120461873512012 | 0.9 | 5.970461873512011 | 6.270461873512012 | 30 | 30 | 0 | 2246 | 2216 | 0.049658887966877195 | 0.9931356184665572 | 0.5319407359621717 | 0.033393501805054154 | 0.01198404360333935 | 0.21198721655009026 | ok |
| 20 | 7.232005076286907 | 0.95 | 7.082005076286906 | 7.382005076286907 | 30 | 30 | 0 | 1791 | 1761 | 0.03962645307911088 | 0.9962105755061175 | 0.5460666480477504 | 0.030096536059057353 | 0.0170740390758092 | 0.2239324081221607 | ok |

## Current Interpretation Checkpoint

- `avgdeg = 6, 10, 20` remain broadly GOE-like across the sampled quantile windows.
- No convincing Poisson-like edge-spacing law appears in the current homogeneous ER runs.
- The main anomaly remains `avgdeg = 3` near `q = 0.50`, `E ≈ 0`, where:
  - `mean_gap_ratio ≈ 0.484`
  - `P(s < 0.2) ≈ 0.301`
  - both KS distances are very large and essentially equal
- This makes the low-degree center behavior the natural next diagnostic target.

## Saved Follow-Up Target: `avgdeg = 3` Center Anomaly

This note is intentionally preserved for later experiments.

In the main Phase C local-by-`E` run, the strongest non-GOE behavior in homogeneous ER appears at the spectral center for:

- average degree `3`
- quantile `q = 0.50`
- energy `E ≈ 0`
- window approximately `[-0.15, 0.15]`

The relevant numerical indicators are:

- `mean_gap_ratio ≈ 0.484`
- `P(s < 0.2) ≈ 0.301`
- `ks_distance_to_goe ≈ 0.285`
- `ks_distance_to_poisson ≈ 0.285`

Interpretation:

- the center window has an unusually large spike near `s = 0`
- it is much less GOE-like than the neighboring windows for the same `avgdeg = 3` run
- this behavior does **not** appear in the corresponding `avgdeg = 6` center window
- therefore it is recorded as a sparse low-degree center anomaly to revisit in later diagnostic experiments

## Extreme Edge Side Experiment

This was added as a separate exploratory side experiment, intentionally kept outside the main Cook-style local-by-`E` workflow in [erdos_renyi_spacing_by_energy.py](../../src/small_world_mobility_edge/erdos_renyi_spacing_by_energy.py).

Implementation file:

- [erdos_renyi_extreme_edge.py](../../src/small_world_mobility_edge/erdos_renyi_extreme_edge.py)

### Motivation

The Phase C local-by-`E` experiment uses bulk/local-density normalization:

```text
s_i(E) = n_analyzed * rho_hat(E) * (lambda_{i+1} - lambda_i)
```

That is appropriate when there is still a local cloud of many eigenvalues around a chosen spectral location `E`.

At the very extreme edge, however, there are only a few ordered levels and the local density becomes small and unstable. So this side experiment asks a different heuristic question:

- if we isolate only the first few nontrivial edge eigenvalues, do the edge gaps look more GOE-like or more Poisson-like after a simple edge-gap normalization?

This is **not** the same as the main Cook experiment, and it is **not** a rigorous Tracy-Widom or edge-universality scaling study. It is only an exploratory diagnostic.

### Methodology

For each ER graph realization:

1. Generate the graph with the same ER model and component handling as the main Phase C experiment.
2. Analyze the largest connected component when `component_mode="largest_connected_component"`.
3. Compute sorted adjacency eigenvalues.
4. Exclude the top Perron outlier by using `bulk_eigenvalues`.
5. Extract the first `k = EDGE_LEVELS` nontrivial eigenvalues at each bulk edge:
   - lower edge:
     ```text
     mu_1 <= mu_2 <= ... <= mu_k
     ```
   - upper edge:
     ```text
     nu_1 <= nu_2 <= ... <= nu_k
     ```
6. Form edge gaps:
   - lower edge:
     ```text
     Delta_j^(lower) = mu_{j+1} - mu_j,   j = 1,...,k-1
     ```
   - upper edge:
     ```text
     Delta_j^(upper) = nu_{j+1} - nu_j,   j = 1,...,k-1
     ```
7. Pool those raw edge gaps across the ensemble on each side separately.
8. Compute the ensemble mean edge gap on that side:
   ```text
   mean_edge_gap = average of pooled edge gaps on that side
   ```
9. Normalize each edge gap by the ensemble mean:
   ```text
   s_j^(edge) = Delta_j / mean_edge_gap
   ```
10. Compare the pooled normalized edge-gap histogram with:
    - GOE Wigner surmise
    - Poisson spacing density

This is the normalization currently used in the extreme-edge file:

```text
normalized edge gap = raw edge gap / ensemble mean edge gap
```

It differs from the main ER local-spacing script, which uses the empirical local-density rescaling:

```text
s_i(E) = n_analyzed * rho_hat(E) * delta_i
```

### Parameter Snapshot For The Extreme Edge Run

Current defaults in [erdos_renyi_extreme_edge.py](../../src/small_world_mobility_edge/erdos_renyi_extreme_edge.py):

```python
N = 5000
ACTIVE_AVERAGE_DEGREES = [3, 6]
NUM_GRAPHS = 30
SEED = 27182
COMPONENT_MODE = "largest_connected_component"
EDGE_LEVELS = 8
NUM_BINS = 30
SMALL_SPACING_THRESHOLD = 0.20
KS_SAMPLE_SIZE = 4096
```

Saved outputs currently also exist for previous runs with `avgdeg = 10` and `20`, so all four degrees are logged below.

### Generated Edge Outputs

Cross-degree summary:

- [extreme_edge_summary_by_degree.png](../../outputs/erdos_renyi_extreme_edge/extreme_edge_summary_by_degree.png)
- [extreme_edge_summary_all.csv](../../outputs/erdos_renyi_extreme_edge/extreme_edge_summary_all.csv)

Per-degree edge figures and CSV summaries:

- [avgdeg_3/extreme_edge_gap_histograms.png](../../outputs/erdos_renyi_extreme_edge/avgdeg_3/extreme_edge_gap_histograms.png)
- [avgdeg_3/extreme_edge_summary.csv](../../outputs/erdos_renyi_extreme_edge/avgdeg_3/extreme_edge_summary.csv)
- [avgdeg_6/extreme_edge_gap_histograms.png](../../outputs/erdos_renyi_extreme_edge/avgdeg_6/extreme_edge_gap_histograms.png)
- [avgdeg_6/extreme_edge_summary.csv](../../outputs/erdos_renyi_extreme_edge/avgdeg_6/extreme_edge_summary.csv)
- [avgdeg_10/extreme_edge_gap_histograms.png](../../outputs/erdos_renyi_extreme_edge/avgdeg_10/extreme_edge_gap_histograms.png)
- [avgdeg_10/extreme_edge_summary.csv](../../outputs/erdos_renyi_extreme_edge/avgdeg_10/extreme_edge_summary.csv)
- [avgdeg_20/extreme_edge_gap_histograms.png](../../outputs/erdos_renyi_extreme_edge/avgdeg_20/extreme_edge_gap_histograms.png)
- [avgdeg_20/extreme_edge_summary.csv](../../outputs/erdos_renyi_extreme_edge/avgdeg_20/extreme_edge_summary.csv)

### Extreme Edge Numerical Output

| average_degree | side | edge_levels | num_graphs | num_gaps | edge_center_mean | mean_raw_gap | mean_normalized_gap | mean_gap_ratio | small_spacing_probability | ks_distance_to_goe | ks_distance_to_poisson |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | lower | 8 | 30 | 210 | -3.9543167227202716 | 0.03078375220829786 | 1.0 | 0.5163782797562454 | 0.07142857142857142 | 0.262983630952381 | 0.12741815476190477 |
| 3 | upper | 8 | 30 | 210 | 3.9475203193181767 | 0.028712795372012392 | 1.0 | 0.520543030967502 | 0.05714285714285714 | 0.19532412574404762 | 0.1313290550595238 |
| 6 | lower | 8 | 30 | 210 | -5.236425301513103 | 0.026382574493951642 | 1.0 | 0.5167250039231427 | 0.03333333333333333 | 0.1711239769345238 | 0.1868745349702381 |
| 6 | upper | 8 | 30 | 210 | 5.230600417829202 | 0.026660064874228333 | 1.0 | 0.520555714567535 | 0.047619047619047616 | 0.20043480282738096 | 0.1684291294642857 |
| 10 | lower | 8 | 30 | 210 | -6.56127950641257 | 0.02166398402770749 | 1.0 | 0.5469718975679727 | 0.047619047619047616 | 0.08314964657738096 | 0.2168759300595238 |
| 10 | upper | 8 | 30 | 210 | 6.554275389791625 | 0.02173681907347048 | 1.0 | 0.5583064301187327 | 0.0380952380952381 | 0.04066452752976191 | 0.21334635416666667 |
| 20 | lower | 8 | 30 | 210 | -9.04149258246829 | 0.023959035862333097 | 1.0 | 0.5082492321246921 | 0.03333333333333333 | 0.07579520089285714 | 0.17645786830357144 |
| 20 | upper | 8 | 30 | 210 | 9.028856774963765 | 0.023284897455992087 | 1.0 | 0.543146707521566 | 0.023809523809523808 | 0.046379743303571426 | 0.2234770275297619 |

### Extreme Edge Interpretation Checkpoint

- This side experiment is exploratory and uses a heuristic edge-gap normalization, not the Cook-style local-density rescaling and not a rigorous edge-universality scaling.
- For `avgdeg = 3`, the edge-gap histograms look more Poisson-leaning than the local bulk-window histograms, but this does **not** by itself establish a true Poisson edge law.
- For `avgdeg = 6`, the extreme-edge diagnostic remains mixed and should be compared only to another run with the same `N`, `NUM_GRAPHS`, and `EDGE_LEVELS`.
- The more reliable main-phase conclusion still comes from the local-by-`E` experiment:
  - `avgdeg = 6, 10, 20` remain broadly GOE-like
  - `avgdeg = 3` shows the main anomaly near the center, not a clean established Poisson edge law

### Output Timestamps Check

The current extreme-edge outputs on disk were written at:

- `avgdeg_3`: `2026-07-07 17:27:13`
- `avgdeg_6`: `2026-07-07 17:32:30`
- `avgdeg_10`: `2026-07-07 17:37:50`
- `avgdeg_20`: `2026-07-07 17:43:06`

So `avgdeg = 6` is already finished and is not still running.
