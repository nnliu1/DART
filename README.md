# DART: Domain-Agnostic Retrieval-based Type annotation

## Environment

#### 1. conda environment
```
conda create --name dart python=3.8
conda activate dart
```


## Install dependencies
```
pip install -r requirements.txt
```

## Data and Model
#### Download the pretrained DART encoder and training dataset
see [`DART/data/README.md`](DART/data/README.md)
and [`DART/model/README.md`](DART/model/README.md)
## usage 
#### 1. set OpenAI API Key

```
export OPENAI_API_KEY="YOUR API KEY"
```

#### 2. run the full DART pipeline
```
bash DART/src/cta/dart.sh
```

## Example results

experiment results can be found in [`DART/exp/results.ipynb`](DART/exp/results.ipynb)
