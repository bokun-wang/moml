

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
python train.py --alg BSpiderBoost --K 3 --lr 0.001 
```
Output (K = 1)
```
MAML: final test error: avg:0.83089674949646, std:0.06704617921778383, time per iteration: avg:1.7294436252149876, std:0.012749860359540038
MOML: final test error: avg:0.2882386228442192, std:0.007682276006187987, time per iteration: avg:2.2721920854899955, std:0.030573277948210978
NASA: final test error: avg:0.3606844516843557, std:0.009851735745197284, time per iteration: avg:17.609719409458286, std:0.24386260924877234
LocalMOML: final test error: avg:0.4423489113152027, std:0.04615012396852497, time per iteration: avg:2.676222678002191, std:0.03402137251354549
BSpiderBoost: final test error: avg:1.2555736875534058, std:0.07955807849999859, time per iteration: avg:4.208972945508682, std:0.0562012073776881
MOML-V2: final test error: avg:0.44458492785692216, std:0.02172642585182246, time per iteration: avg:8.17596904498994, std:0.08225160968588015
```
Output (K = 3)
```
MAML: final test error: avg:0.30654988989233967, std:0.023351960209257627, time per iteration: avg:1.7812924703899937, std:0.029267328429030483
MOML: final test error: avg:0.16307387098670006, std:0.0050643603905929, time per iteration: avg:2.4841010766200227, std:0.05955570711835525
NASA: final test error: avg:0.2142468300834298, std:0.0495582594735672, time per iteration: avg:18.10059960091683, std:0.1945554632168004
LocalMOML: final test error: avg:0.1610884727910161, std:0.018253115165757026, time per iteration: avg:2.846339862552933, std:0.09168362043915836
BSpiderBoost: final test error: avg:0.6493364048004151, std:0.03293783091266613, time per iteration: avg:4.31338609815545, std:0.019607381335747753
MOML-V2: final test error: avg:0.27341685578227043, std:0.016500978241562977, time per iteration: avg:8.195019181519992, std:0.013678508239016786
```