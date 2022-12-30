Download the preprocessed data from [here](https://drive.google.com/drive/folders/1hiEtsVDrKNk1qqHz2keSpXIGWHLwuUOn?usp=sharing) and put the files in the './data' folder.

#### Main file for running Per-FedAvg on CIFAR-100
```
import sys

sys.path.append("..")
import train
import os

train.config["rank"] = int(os.environ['SLURM_PROCID'])

train.config["beta"] = 1.0
train.config["total_iters"] = 10000
train.config["H"] = 4
train.config["n_workers"] = 4
train.config["ft_lr"] = 0.001
train.config["update_lr"] = 0.001
train.config["meta_lr"] = 0.001
train.config["k_spt"] = 5
train.config["seed"] = 0
train.config["num_tasks"] = 1
train.config["dataset"] = "cifar100"
train.config["n_way"] = 1
train.config["distributed_init_file"] = "./output/dist_init"

train.main()
```
#### Main file for running LocalMOML on CIFAR-100
```
import sys

sys.path.append("..")
import train
import os

train.config["rank"] = int(os.environ['SLURM_PROCID'])

train.config["beta"] = 0.9
train.config["total_iters"] = 10000
train.config["H"] = 4
train.config["n_workers"] = 4
train.config["ft_lr"] = 0.001
train.config["update_lr"] = 0.001
train.config["meta_lr"] = 0.001
train.config["k_spt"] = 5
train.config["seed"] = 0
train.config["num_tasks"] = 1
train.config["dataset"] = "cifar100"
train.config["n_way"] = 1
train.config["distributed_init_file"] = "./output/dist_init"

train.main()
```

#### Main file for running pFedMe on CIFAR-100
```
import sys

sys.path.append("..")
import train_pfedme as train
import os

train.config["rank"] = int(os.environ['SLURM_PROCID'])

train.config["local_steps"] = 10
train.config["lam"] = 500
train.config["total_iters"] = 10000
train.config["H"] = 4
train.config["n_workers"] = 4
train.config["update_lr"] = 0.001
train.config["ft_lr"] = 0.001
train.config["meta_lr"] = 0.001
train.config["k_spt"] = 5
train.config["seed"] = 0
train.config["num_tasks"] = 1
train.config["dataset"] = "cifar100"
train.config["n_way"] = 1
train.config["distributed_init_file"] = "./output/dist_init_pfedme"

train.main()
```

#### The distributed experiment can be launch by

```
srun python main.py
```