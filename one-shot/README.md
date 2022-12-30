Download the preprocessed data from [here](https://drive.google.com/drive/folders/1SNpJig1EaywMBN56eGg7LQde0AFVykd3?usp=sharing) and put the files in the './data' folder.

#### 1) Omniglot
```
python train.py —alg MAML —dataset Omniglot —lr=0.001
python train.py --alg MOML --dataset Omniglot --lr=0.001 --beta 0.3
python train.py --alg LocalMOML --dataset Omniglot --lr=0.001 --beta 0.3
python train.py --alg MOML-V2 --dataset Omniglot --lr=0.001 --beta 0.5
python train.py —alg ProtoNet —dataset Omniglot —lr=0.1
python train.py —alg Reptile —dataset Omniglot —lr=0.001
python train.py —alg NASA —dataset Omniglot —lr=0.05
```
#### 2) Cifar-100
```
python train.py --alg MAML --dataset CIFAR-100 --lr=0.001
python train.py --alg MOML --dataset CIFAR-100 --lr=0.001 --beta 0.3
python train.py --alg LocalMOML --dataset CIFAR-100 --lr=0.001 --beta 0.7
python train.py --alg MOML-V2 --dataset CIFAR-100 --lr=0.001 --beta 0.5
python train.py --alg ProtoNet --dataset CIFAR-100 --lr=0.05
python train.py --alg Reptile --dataset CIFAR-100  --lr=0.01
python train.py --alg NASA --dataset CIFAR-100 --lr=0.1 --beta 0.3 
```
Output (Omniglot)
```
MAML: final test accuracy (percent): avg:44.311666666666696, std:1.2936661942797358
MOML: final test accuracy (percent): avg:46.34500000000003, std:1.3798369468890186
LocalMOML: final test accuracy (percent): avg:46.235000000000035, std:1.7016217754444323
MOML-V2: final test accuracy (percent): avg:45.81333333333337, std:0.31943526556861335
ProtoNet: final test accuracy (percent): avg:45.66666793823242, std:3.299830675125122
Reptile: final test accuracy (percent): avg:46.02333333333337, std:1.082314905910275
NASA: final test accuracy (percent): avg:46.146666666666704, std:0.8197187864681907
```
Output (Cifar-100)
```
MAML: final test accuracy (percent): avg:40.17777777777781, std:0.4878777422244501
MOML: final test accuracy (percent): avg:40.488888888888916, std:0.20608041101101057
LocalMOML: final test accuracy (percent): avg:40.37777777777781, std:0.7350149070462757
MOML-V2: final test accuracy (percent): avg:40.82222222222225, std:0.25141574442188436
ProtoNet: final test accuracy (percent): avg:31.111116409301758, std:6.285393714904785
Reptile: final test accuracy (percent): avg:40.08888888888892, std:0.11331154474650965
NASA: final test accuracy (percent): avg:40.60000000000003, std:0.33110365390558655
```