# Dependencies
this project is dependent on ShrinkBench, for proper instalation follow their instructions at https://github.com/SonyPony/shrinkbench/tree/47dd89ee4d44fefd34c2c900a23b4f3dd0ac785c

# Lottery ticket na MNISTnet
usage:

```
find_lt_mnist.py -d DATA -m MODEL [--epochs EPOCHS] [--p_rate P_RATE] [--p_perc P_PERC] [--lr LR ] [--strategy STRATEGY]
```

DATA - path to training data folder
MODEL - path to trained model weights
EPOCHS - desired number of epochs 
P_RATE - prunning rate on each prunning iteration
P_PERC - desired final prunning percentage
LR - learning rate
STRATEGY - prunning strategy [GlobalMagWeight|LayerMagWeight]

# Lottery ticket on Tiny ImageNet
usage:

```
find_lt_tinyimagenet.py -d DATA -m MODEL [--epochs EPOCHS] [--p_rate P_RATE] [--p_perc P_PERC] [--lr LR ] [--strategy STRATEGY]
```

DATA - path to training data folder
MODEL - path to trained model weights
EPOCHS - desired number of epochs 
P_RATE - prunning rate on each prunning iteration
P_PERC - desired final prunning percentage
LR - learning rate
STRATEGY - prunning strategy [GlobalMagWeight|LayerMagWeight]
