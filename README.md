# NeuroPath

## Setup
```shell
conda create -n neuropath python==3.11
conda activate neuropath
```

It is recommended to install `torch` first. As specified in our reproducibility requirements, version `2.4.0` should be used.

The following is an example of torch installation under CUDA 12.4. Please install according to your actual device requirements.
```shell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

Install other dependencies.
```shell
# Install faiss-gpu
conda install -c pytorch -c nvidia faiss-gpu=1.9.0

# Install requirements.txt
pip install -r requirements.txt 
```

## Reproduction

### Datasets
All datasets used in the experiments are stored in the `data` directory, including three multi-hop QA datasets and two simple QA datasets.

### Optional environment variables configuration
```shell
export HF_HOME="your_hf_home_path"
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url" # default: https://api.openai.com/v1

# To avoid network issues when connecting to Hugging Face, we recommend downloading the model in advance and using it in local mode.
export TRANSFORMERS_OFFLINE=1  # Optional 
```

### Run NeuroPath
Please first check the script content and specify a specific dataset for the experiment. Specific parameters can be set in the script.
#### Indexing
```shell
bash src/setup_neuropath_main_exp.sh
```
#### Retrieval
```shell
bash src/run_neuropath_main_exp.sh
```


#### QA
If you want to perform QA, refer to the `README.md` in `src\qa` directory


### Baselines
Please refer to the README.md file in the `src/baselines` directory for detailed instructions.

>  Note: If you need to use BM25, you must install ElasticSearch separately and uncomment the line from elasticsearch import Elasticsearch in the relevant code.  We recommend using elasticsearch==9.0.1.

For HippoRAG, LightRAG, and PathRAG, please refer to their respective repositories.