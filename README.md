## Selective layer tuning and performance study of pre-trained models using genetic algorithm with single-channel dataset

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

[//]: # (![Preprocessing audio dataset]&#40;img/img.png&#41;)
<img src="imgs/Figure 2.png" alt="J" width="720"/>

### ./G_A_based:

* `G_A_EfficientNet.py`: Selective layer tuning based on genetic algorithm in Pre-trained EfficientNet models.
* `G_A_MobileNet.py`: Selective layer tuning based on genetic algorithm in Pre-trained MobileNet model.
* `G_A_ResNet_3types.py`: Selective layer tuning based on genetic algorithm in Pre-trained ResNet models.
* `G_A_VGG.py`: Selective layer tuning based on genetic algorithm in Pre-trained VGG model.

### Detailed "Selective layer tuning based on genetic algorithm":

[//]: # (![Detailed Selective layer tuning based on genetic algorithm]&#40;img/img.png&#41;)
<img src="imgs/Figure 4.png" alt="J" width="720"/>

[//]: # (![Detailed Selective layer tuning based on genetic algorithm]&#40;img/img.png&#41;)
<img src="imgs/Figure 5.png" alt="J" width="720"/>

* Genome: The gene is a pre-trained model trained with the ImageNet dataset that allows all layers to be selected as a trainable layer and a freezing layer.
* Initial generation: In the first generation, the entire layer for each genome is randomly selected as the trainable layer and freezing layer.
* Fitness evaluation: Short training is performed on a given dataset based on randomly selected trainable and freezing layers, and validation accuracy is obtained for each epoch. The highest fitness score is ranked by setting a validation acuity as a fitness indicator.
* Selection: Make a choice from the top dominant genomes selected through fitness evaluation.
* Crossover: Select and cross over the selected dominant genomes. Two main crossover methods are used, as shown in Figure 5.
* Mutation: In all genomes of the current generation, a layer is randomly selected with a 4% probability and reversed.
* Next generation: Half of all genomes are selected as dominant genomes, and the dominant genomes are converted into child genomes using two crossover methods. All selected genomes undergo mutation. Finally, the total number of dominant and child genomes will be the same as the initial number of populations in the next generation.
* Iteration: Selection, crossover, mutation, and next generation are repeated until the target is achieved.

### Experimental results:

[//]: # (![Experimental results about MNIST-Fashion]&#40;img/img.png&#41;)
<img src="imgs/Table 3.png" alt="J" width="720"/>

[//]: # (![Experimental results about UrbanSound8K]&#40;img/img.png&#41;)
<img src="imgs/Table 4.png" alt="J" width="720"/>

In the experiment on grayscale and Mel-spectrogram images, selective layer tuning based on a genetic algorithm showed better performance than the heuristic tuning or fine-tuning of the pre-trained model, or tuning the selective layer at random. In addition, selective layer tuning based on a genetic algorithm can obtain high performance while minimizing the number of trainable parameters required for fine-tuning a pre-trained model. Thus, given a pre-trained model suitable for various tasks, a high-performance tuned pre-trained model according to the data can be easily and conveniently obtained.

### Example of selective layer tuning based on genetic algorithm in Pre-trained VGG model:

[//]: # (![MNIST-Fashion dataset-based selective layer tuning by VGG16]&#40;img/img.png&#41;)
<img src="imgs/Figure 7.png" alt="J" width="720"/>

[//]: # (![UrbanSound8K dataset-based selective layer tuning by VGG16]&#40;img/img.png&#41;)
<img src="imgs/Figure 8.png" alt="J" width="720"/>

## Datasets

#### This repository is using 3 datasets:

* Fashion-MNIST dataset can be found at https://github.com/zalandoresearch/fashion-mnist.
* UrbanSound8K dataset can be found at https://urbansounddataset.weebly.com/urbansound8k.html.
* HospitalAlarmSound dataset is available online. If you need this dataset, call by email via sayney1004@gmail.com