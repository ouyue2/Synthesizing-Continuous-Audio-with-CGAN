# Synthesizing Continuous Audio with CGAN

## Motivation
In our life, it is very useful to play natural sounds like bird songs, ocean wave, flowing river, etc. But most of present works are only able to repeat a audio clip again and again. Therefore, we propose a generative adversarial network to synthesizing audio continuously, based on [WaveGAN](https://github.com/chrisdonahue/wavegan). 

## Requirement
```
conda install tensorflow-gpu==1.12.0
conda install scipy==1.0.0
conda install matplotlib==3.0.2
conda install -c conda-forge librosa==0.6.2
```

## Data Selection

## Training

### Backup
Sometimes training may occasionally collapse, so it is highly recommended to back up checkpoints offen. 
To back up checkpoints in default path ```./train``` every hour, use
```
python backup.py
```
Set ```--train_dir``` and ```--backup_time``` to customize the backups.

### Monitor
To monitor training via Tensorboard, use
```
tensorboard --logdir=./train --port YOUR_PORT
```

## Generating
