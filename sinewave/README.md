

## Sinewave regression experiment

### Dependencies
- Python 3.6+
- Pytorch

### Supported algorithms

See the '.src' folder.

- MAML
- MOML (v1) and MOML-v2
- LocalMOML
- NASA
- BSpiderBoost

### Scripts for reproducing the reported results

The stepsize is tuned in {0.1, 0.05, 0.01, 0.005, 0.001}. The moving average parameter beta is tuned in {0.1, 0.3, 0.5, 0.7, 0.9} for MOML variants.

#### 1) K = 1

```
python train.py --alg MAML --K 1 --lr 0.001
python train.py --alg MOML --K 1 --lr 0.005 --beta 0.3
python train.py --alg NASA --K 1 --lr 0.05 --beta 0.5 --grad_mom 0.1
python train.py --alg MOML-V2 --K 1 --lr 0.005 --beta 0.1
python train.py --alg LocalMOML --K 1 --lr 0.005 --beta 0.3 --K0 2 
python train.py --alg BSpiderBoost --K 1 --lr 0.001 
```
#### 2) K = 3

```
python train.py --alg MAML --K 3 --lr 0.005
python train.py --alg MOML --K 3 --lr 0.01 --beta 0.5
python train.py --alg NASA --K 3 --lr 0.05 --beta 0.5 --grad_mom 0.1
python train.py --alg MOML-V2 --K 3 --lr 0.01 --beta 0.1
python train.py --alg LocalMOML --K 3 --lr 0.01 --beta 0.5 --K0 6
python train.py --alg BSpiderBoost --K 3 --lr 0.005 
```
Output (K = 1)
```
MAML: final test error: avg:0.835590249300003, std:0.07186732750571577, time per iteration: avg:2.5864917459135324, std:0.05509648220147861
MOML: final test error: avg:0.2904560447484255, std:0.0069210130475210905, time per iteration: avg:3.3251332053290916, std:0.12252181036195989
NASA: final test error: avg:0.3652022566646338, std:0.020095324862615283, time per iteration: avg:24.924140069050274, std:2.1450915546867915
LocalMOML: final test error: avg:0.43164974026381964, std:0.052984637636829486, time per iteration: avg:4.268289895636097, std:0.455377787667406
BSpiderBoost: final test error: avg:1.2754727530479433, std:0.10192635067390943, time per iteration: avg:5.763388980216599, std:0.06985328101127464
MOML-V2: final test error: avg:0.4485846327245236, std:0.02519443646586023, time per iteration: avg:9.287330184210662, std:0.3708868472516213
```
Output (K = 3)
```
MAML: final test error: avg:0.30827567011117935, std:0.027673786484705743, time per iteration: avg:3.2857486592789393, std:0.15374948716297965
MOML: final test error: avg:0.17320605453103782, std:0.039194603400819504, time per iteration: avg:3.2785870247595468, std:0.04099428707934505
NASA: final test error: avg:0.19471208553761243, std:0.02113302921625866, time per iteration: avg:24.807282468735405, std:0.6065908816543528
MOML-V2: final test error: avg:0.26830849956721065, std:0.025188366251520013, time per iteration: avg:8.998131329849791, std:0.4024166749866838
LocalMOML: final test error: avg:0.15732655223459005, std:0.008396704140571856, time per iteration: avg:3.922237055682744, std:0.1015244640764435
BSpiderBoost: final test error: avg:0.5014390408992767, std:0.08673397277402035, time per iteration: avg:6.267515634516499, std:0.03693094398223238
```