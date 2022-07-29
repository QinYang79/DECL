
PyTorch implementation for [Deep Evidential Learning with Noisy Correspondence for Cross-modal Retrieval](https://drive.google.com/file/d/1YVXD2ki5txBY6khG62EHwCi6cnQVRE4I/view) (ACM Multimedia 2022).


## Introduction

### DECL framework
<img src="src/framework.png"  width="860" height="268" />

## Requirements

- Python 3.8
- PyTorch (>=1.10.0)
- numpy
- scikit-learn
- TensorBoard
- Punkt Sentence Tokenizer:
  
```
import nltk
nltk.download()
> d punkt
```
  
## Datasets

Our directory structure of ```data```.
```
data
├── f30k_precomp # pre-computed BUTD region features for Flickr30K, provided by SCAN
│     ├── train_ids.txt
│     ├── train_caps.txt
│     ├── ......
│
├── coco_precomp # pre-computed BUTD region features for COCO, provided by SCAN
│     ├── train_ids.txt
│     ├── train_caps.txt
│     ├── ......
│
├── cc152k_precomp # pre-computed BUTD region features for cc152k, provided by NCR
│     ├── train_ids.txt
│     ├── train_caps.tsv
│     ├── ......
│   
├── noise_file # Randomly shuffle the index of the image proportionally.
│     ├── f30k
│     │     ├── noise_inx_0.2.npy
│     │     ├── ......
│     │ 
│     └── coco
│           ├── noise_inx_0.2.npy
│           ├── ......     
│
└── vocab  # vocab files provided by SCAN and NCR
      ├── f30k_precomp_vocab.json
      ├── coco_precomp_vocab.json
      └── cc152k_precomp_vocab.json
```

### MS-COCO and Flickr30K
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies.

### CC152K
Following [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR), we use a subset of [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions) (CC), named CC152K. CC152K contains training 150,000 samples from the CC training split, 1,000 validation samples and 1,000 testing samples from the CC validation split.

[Download Dataset](https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar)

### Noise index
If you want to experiment with the same noise index as in the paper, the noise index files can be downloaded from [here](https://drive.google.com/file/d/1-PJTDZRMo68mtY-hzPXheaakOVRkY5Ie/view?usp=sharing).


## Training and Evaluation

### Training new models
Modify some necessary parameters (i.e., ```data_path```, ```vocab_path```, ```noise_ratio```, ```warmup_epoch```, ```module_name```, and ```folder_name``` ) in ```train_xxx.sh``` file and run it.

For Flickr30K:
```
sh train_f30k.sh
```

For MSCOCO:
```
sh train_coco.sh
```

For CC152K:
```
sh train_cc152k.sh
```

### Evaluation
Modify the  ```data_path```, ```vocab_path```, ```checkpoint_paths``` in the ```eval.py``` file and run it.
```
python eval.py
```

Our reproduced results in [evaluation_log](https://drive.google.com/file/d/1N14yx5YE6kT1h9TvcJi8w3ALlSl3TbUc/view?usp=sharing). (Better than the original paper)

### Experiment Results:
<img src="./src/tab1.png"  width="740" />
<img src="./src/tab2.png"  width="740" />


## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR), [SGRAF](https://github.com/Paranioar/SGRAF), and [SCAN](https://github.com/kuanghuei/SCAN) licensed under Apache 2.0.
