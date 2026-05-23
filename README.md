# README

## Project Introduction

This is the implementation for the COMP7250 2026 course project at Hong Kong Baptist University\. For the code clone detection task, we implemented a data flow\-aware CodeBERT model\.
Based on the pre\-trained CodeBERT, we introduce data flow dependencies between code variables to customize dynamic attention masks, enabling the model to better capture the semantic structure information of code, thereby improving the performance of clone detection\.

## Core Features

- **Data Flow\-Aware Attention Mask**: Based on the data flow dependencies between variables in the code, we restrict the scope of attention calculation, allowing only variables with data dependencies to perform attention interaction, enhancing the models understanding of code semantics\.

- **Custom Position Embedding**: We design special position ids for variable tokens to adapt to the position encoding range of CodeBERT, solving the position encoding problem of long sequences\.

- **Efficient Data Preprocessing**: Based on Tree\-sitter, we implement AST parsing for Java code, automatically extract variables and data flow edges, and complete the preprocessing of the dataset\.

- **Mixed Precision Training**: We use PyTorch AMP to implement mixed precision training, reducing memory usage and improving training speed\.

## Environment Dependencies

First install the required Python dependencies:

```bash
pip install torch transformers datasets tree-sitter tree-sitter-java numpy tqdm pandas
#This program runs on a server with an NVIDIA A100 40G.
```

## Project Structure

```Plain
.
├── data_process.py          # Dataset preprocessing script, processes train/validation/test sets
├── main.py                  # Main script for model training and evaluation, includes custom model implementation
└── test.py                  # Test script for data flow extraction function, to verify the parsing function

```

## Quick Start
### 1\. Function Test

Before starting to train, if you want to verify whether the data flow extraction function works properly, you can run the test script:

```bash
python test.py
```

This script uses a sample Java code to test the parsing capability of Tree\-sitter, outputs the extracted variable list and data flow edges, to verify whether the parsing function works normally\.

### 2\. Data Preprocessing

First run the data preprocessing script\. This script will automatically download the BigCloneBench dataset, complete data flow extraction, input formatting and other preprocessing work, and save the processed dataset locally:

```bash
python data_process.py
```

> Notes:
> 
> - This step will automatically download a \~100MB dataset
> 
> - After processing, three directories `processed_train`, `processed_val`, `processed_test` will be generated to store the preprocessed dataset
> 
> 

### 3\. Model Training \& Evaluation

After preprocessing, run the main script to start the training and evaluation process:

```bash
python main.py
```

Training \& Evaluation Process:

- The script will automatically detect and use GPU for training \(if CUDA is available\)

- After each training epoch, the model will automatically be evaluated on the validation set

- The best\-performing model on the validation set will be automatically saved to `checkpoints/best_model.pth`

- Training logs \(training loss, gradient norm, validation accuracy, learning rate, etc\.\) will be saved to `checkpoints/training_log.csv`

- After all training is completed, the best model will be automatically loaded for final evaluation on the test set, and the test set accuracy will be output\.


## Configuration Adjustment

You can modify the configuration parameters at the top of `main.py` to adapt to your hardware environment and training requirements:

```python
MAX_SEQ_LEN = 448        # Maximum input sequence length
BATCH_SIZE = 4           # Training batch size, you can reduce it if GPU memory is insufficient
EPOCHS = 3               # Number of training epochs
LEARNING_RATE = 2e-5     # Initial learning rate
WARMUP_RATIO = 0.1       # The ratio of warmup steps to total training steps
WEIGHT_DECAY = 0.01      # Weight decay coefficient
```

## Dataset

This project uses the **BigCloneBench** dataset from the CodeXGlue benchmark, which is a standard dataset for the code clone detection task\. It contains a large number of Java code function pairs, used to determine whether two pieces of code are clone code\.

## Acknowledgements

- Pre\-trained Model: [microsoft/codebert\-base](https://huggingface.co/microsoft/codebert-base)

- Dataset: [CodeXGlue BigCloneBench](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench)

- Code Parsing: [Tree\-sitter](https://github.com/tree-sitter/tree-sitter)
