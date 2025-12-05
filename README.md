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
conda activate unisplicer   # or the environment name generated
```

## Workflow
### Step 1 Training Data Preparation

Extract genomic regions and reference splice sites from genome and annotation files.

### Step 2 Generate Training Data

Two pathways are provided
GCF dataset processing
GCA dataset processing with an included example
Both produce formatted donor and acceptor training datasets.

### Step 3 Model Training

Run the training script
```bash
python Step3_UniSplicer_model_training_source_code.py --species Arabidopsis_thaliana --batchsize 32 --cnn_hidden_unit 60 --lstm_hidden_unit 60 --lstm_layer_num 3 --window_context 600 --epoch_number 10 --lr_rate 1e-3 --lossweight 10.0&
```

### Step 4 Model Evaluation
Compute Top K accuracy using the evaluation notebook.
Supports cross species evaluation and mutation effect scoring.

## Usage Notes

UniSplicer can be used for

Splice site prediction

Analysis of splice altering mutations

Training custom models for new species

Comparative genomics and evolutionary studies


## Citation

If you use this software, pipeline, or trained models for academic research or publications, please cite the following paper:

Conghao Hong, Wenzhen Cheng, Zhengyi Li, Jiajie Deng, Yiqiong Li, Youyi Zang, and Hongbo Gao. "UniSplicer deep learning models across diverse taxa for highly accurate intron splice site prediction and splice altering mutation detection" In press 2025

## License and Usage Policy

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
