# Semantic Orthogonal Activation Steering: A Tuning-Free Defense for Large Language Models Against Data Extraction (SOAS)
The code for our paper "Semantic Orthogonal Activation Steering: A Tuning-Free Defense for Large Language Models Against Data Extraction"

## About The Project
**SOAS** is a simple yet effective tuning-free defense mechanism against data extraction attacks targeting the training data of LLMs. It identifies critical layers enabling successful extraction by analyzing activations between successive layers, calculates a steering vector orthogonal to the semantic direction of the protected data, and applies this vector to the identified layer, yielding effective defense against existing data extraction attacks.

## Getting Started
### File Structure 
```
SOAS-code/
├── utils
│   ├── config_for_all.py
│   ├── my_util.py
│   ├── st_generation.py
│   ├── st_deployment.py
│   └── evaluation.py
└── main.py
```
The codebase consists of several parts:

- `config_for_all.py`: It contains the settings for various hyperparameters.
- `my_util.py`: It includes preprocessing of raw datasets, loading of victim models, and calculation of evaluation metrics.
- `st_generation.py`: It contains functions for defense layer selection and steering vector generation.
- `st_deployment.py`: It includes functions for steering vector deployment.
- `evaluation.py`: It contains evaluations of generated suffixes protected by SOAS.
- `main.py`: The main function of **SOAS**. 

### Requirements

* python 3.10.15 
* [pytorch](https://pytorch.org/get-started/locally/) 2.1.0 + cu118
* CUDA 12.6 and above are recommended
* transformers 4.39.3
* accelerate 1.1.0
* tokenizers 0.20.0
* evaluate 0.4.1

Before running the project, ensure you have the correct environment and required packages installed.

### Hyper-parameters 
The settings of **SOAS** are determined in the parameter **args** in **config_for_all.py**. Here, we mainly introduce the important hyper-parameters.
- model: the root of victim model. Default:"gpt-j-6b".
- prefix_root: the root of ground-truth prefixes. Default: 50.
- suffix_root: the root of ground-truth suffixes. Default: 50.
- chunk: the number of samples for evaluation. Default: 1500.
- prefix_root: the tested token length of ground-truth prefixes. Default: 50.
- suffix_root: the tested token length of ground-truth suffixes. Default: 50.
- steering strength: the coefficient to control the intensity of SOAS. Default: 10.0.
- bs: the batch size for all experiments. Default: 8.

### Run
You could run `main.py` in your python IDE directly.
The example codes below show the workflow to perform a complete fingerprinting process, which is in `main.py`.

```python
def main(args):
    # loading model
    model, tokenizer, num_of_layers, emb_dim, projection_matrix = get_model()

    # defense layer selection
    defense_layer_idx, _ = defense_layer_selection(args, model, num_of_layers, projection_matrix)

    # loading steering vector
    steering_vec = steering_vector_generation(args, model, tokenizer, num_of_layers, emb_dim)

    # steering vector generation
    steering_vector_deployment(args, defense_layer_idx)

    # defense evaluation
    defense_evaluation(args, model, tokenizer)
```

You can also run main.py using the cmd command.

```python
$ python main.py --model "gpt-j-6b" --chunk 1500 --steering strength 10.0 --bs 8
```

## Note
- The protected dataset utilized in our paper can be found [here](https://github.com/google-research/lm-extraction-benchmark).
- All the downstream datasets and relevant evaluations can be found [here](https://github.com/joeljang/knowledge-unlearning/tree/main/validation_data). 
- The implementations of existing data extraction attacks can be found [here](https://github.com/ftramer/LM_Memorization/tree/main).
