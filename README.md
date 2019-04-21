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
For GUI
```
conda install pyqt==5.9.2
```

## Dataset
You can use any folders containing audio for training. Here are some examples:
- [Bird Song](http://people.tamu.edu/~ouyue2/files/s_gen/data/bird.wav), from [Forest Birdsong](https://www.youtube.com/watch?v=Qm846KdZN_c), checkpoints for reloaded or generating available on [Google Drive](https://drive.google.com/drive/folders/1tUqYkWs_mxOduUx7-uz3-JCyHIKicOjD?usp=sharing)
- [Ocean Wave](http://people.tamu.edu/~ouyue2/files/s_gen/data/ocean.wav), from [Relaxing Video of A Tropical Beach](https://www.youtube.com/watch?v=qREKP9oijWI)
- [Piano](http://people.tamu.edu/~ouyue2/files/s_gen/data/piano.wav)(Not Recommeded), from [Beautiful Piano Music](https://www.youtube.com/watch?v=HSOtku1j600)
- [Railway Train](http://people.tamu.edu/~ouyue2/files/s_gen/data/railwaytrain.wav), from [Train Sounds](https://www.youtube.com/watch?v=R-R65Gg0CJ8)

Example results from datasets above are available on [Google Drive](https://drive.google.com/drive/folders/12m0RmGZlqs3sw-aXUNP0nfvdK5cuPA6K?usp=sharing). 

## Training
To train data from data in ```./data/```, use
```
python main.py train --data_dir ./data/ --data_fast_wav --verbose
```

### Backup
Sometimes training may occasionally collapse, so it is highly recommended to back up checkpoints often. 
To back up checkpoints in default path ```./train/``` every hour, use
```
python backup.py
```
Set ```--train_dir``` and ```--backup_time``` to customize the backups.

### Monitor
To monitor training via Tensorboard, use
```
tensorboard --logdir ./train --port PORT
```

## Generating

### Generating in Command Line 
To generate audio in command line, use
```
python main.py generate --wav_out_time 150
```
Generating will use the latest checkpoint in train dir unless ```--ckpt_path``` is set to your model path. ```--wav_out_time``` should be set in seconds, omitting means generating until KeyboardInterrupt called. 

### Generating with GUI
To run GUI for generating audio, use
```
python gui_generate.py
```
Parameters can be set in GUI. 
