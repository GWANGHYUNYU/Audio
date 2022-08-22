## This is genetic algorithm for searching hyper-parameters in pre-trained model

## Requirements
Installation the list of libraries
1) tensorflow2
2) numpy
3) matplotlib
4) random
5) deepcopy
6) pickle
7) time
8) librosa

## Structure of repository
```
.
├── DSP
│   ├── cut_audio.ipynb
│   ├── cut_audio2.ipynb
│   ├── dsp1.ipynb
│   ├── dsp2.ipynb
│   ├── dsp3.ipynb
│   ├── dsp4.ipynb
│   ├── preprocessing_Audio.ipynb
│   └── preprocessing_HospitalSound.ipynb
├── GA
    ├── G_A_based_audio
    │   ├── G_A_EfficientNet.py
    │   ├── G_A_MobileNet.py
    │   ├── G_A_ResNet_3types.py
    │   └── G_A_VGG.py
    ├── G_A_based_mnistfashion
    │   ├── G_A_EfficientNet.py
    │   ├── G_A_MobileNet.py
    │   ├── G_A_ResNet_3types.py
    │   └── G_A_VGG.py
    └── G_A_based_urbansound
        ├── G_A_EfficientNet.py
        ├── G_A_MobileNet.py
        ├── G_A_ResNet_3types.py
        ├── G_A_VGG.py
        ├── load_file.ipynb
        └── test_urbansound8k.ipynb

```

## Detailed load data

* `preprocessing_HospitalSound.ipynb`: Read all HospitalAlarmSound files, then preprocess the Log_Mel_spectrogram.

* `load_file.ipynb`: Read all UrbanSound8K files, then preprocess the Log_Mel_spectrogram.