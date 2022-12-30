

## Sinewave regression experiment

### Dependencies
- Python 3.6+
- Pytorch

### Supported algorithms

See the '.src' folder.

- MAML
- Reptile
- MOML (v1) and MOML-v2
- LocalMOML
- NASA
- BSpiderBoost

### Scripts for reproducing the reported results

The stepsize is tuned in {0.1, 0.05, 0.01, 0.005, 0.001}. The moving average parameter beta is tuned in {0.1, 0.3, 0.5, 0.7, 0.9} for MOML variants.

#### 1) K = 1

```
python train.py --alg MAML --K 1 --lr 0.001
python train.py --alg Reptile --K 1 --lr 0.05
python train.py --alg MOML --K 1 --lr 0.005 --beta 0.3
python train.py --alg NASA --K 1 --lr 0.05 --beta 0.5 --grad_mom 0.1
python train.py --alg MOML-V2 --K 1 --lr 0.005 --beta 0.1
python train.py --alg LocalMOML --K 1 --lr 0.005 --beta 0.3 --K0 2 
python train.py --alg BSpiderBoost --K 1 --lr 0.001 
```
#### 2) K = 3

```
python train.py --alg MAML --K 3 --lr 0.005
python train.py --alg Reptile --K 3 --lr 0.1
python train.py --alg MOML --K 3 --lr 0.01 --beta 0.5
python train.py --alg NASA --K 3 --lr 0.05 --beta 0.5 --grad_mom 0.1
python train.py --alg MOML-V2 --K 3 --lr 0.01 --beta 0.1
python train.py --alg LocalMOML --K 3 --lr 0.01 --beta 0.5 --K0 6
python train.py --alg BSpiderBoost --K 3 --lr 0.001 
```
