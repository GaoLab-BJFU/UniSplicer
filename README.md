# UniSplicer

UniSplicer is a universal deep learning framework for predicting intron splice sites with high accuracy across diverse taxa.
It supports training species specific and multi species models, identifying splice altering mutations, and evaluating model performance with Top K accuracy.

This repository includes complete workflows for data preprocessing, training dataset generation, model training, and accuracy evaluation.


## Installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/GaoLab-BJFU/UniSplicer.git
cd UniSplicer
conda env create -f environment.yml
conda activate unisplicer
```

Install ASTool which is required for alternative splicing related preprocessing
For the detailed installation process, please refer to
https://github.com/zzd-lab/ASTool


## Workflow

### Step 1. Training Data Preparation

Extract intron splice sites detected by RNA-seq.

### Step 2. Generate Training Data

Two example codes are provided:
GCF dataset processing code,
GCA dataset processing code

### Step 3. Model Training

Run the training script
```bash
python Step3_UniSplicer_model_training_source_code.py --species Arabidopsis_thaliana --batchsize 32 --cnn_hidden_unit 60 --lstm_hidden_unit 60 --lstm_layer_num 3 --window_context 600 --epoch_number 10 --lr_rate 1e-3 --lossweight 10.0
```
If you are using transfer learning:

make sure you have the base UniSplicer models first, and then add the transfer learning flag:"--enable_transfer_learning"
```bash
python Step3_UniSplicer_model_training_source_code.py --species Rosa_chinensis --batchsize 32 --cnn_hidden_unit 60 --lstm_hidden_unit 60 --lstm_layer_num 3 --window_context 600 --epoch_number 10 --lr_rate 1e-3 --lossweight 10.0 --enable_transfer_learning
```

### Step 4. Model Evaluation
Compute Top-K accuracy using the evaluation notebook.


## Usage Notes

UniSplicer can be used for:

(1) Splice site prediction,

(2) Analysis of splice altering mutations,

(3) Training custom models for new species,

(4) Comparative genomics and evolutionary studies.


## Citation

If you use this software, pipeline, or trained models for academic research or publications, please cite the following paper:

Conghao Hong, Wenzhen Cheng, Zhengyi Li, Jiajie Deng, Yiqiong Li, Youyi Zang, and Hongbo Gao. "UniSplicer deep learning models across diverse taxa for highly accurate intron splice site prediction and splice altering mutation detection" In press 2025

## License and Usage

This software is free for academic research use and personal use.
Commercial use is not permitted.

By using this repository or any part of the UniSplicer code, you agree to the following

No commercial redistribution or commercial application is allowed


## Contact

For questions or collaboration inquiries, please contact: 

College of Biological Sciences and Technology, Beijing Forestry University, Beijing 100083 China

Correspondence

Hongbo Gao

Email gaohongbo@bjfu.edu.cn
