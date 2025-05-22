# Baselines

## Base Retriever (Naive)

### BM25

Before using BM25, ElasticSearch should be installed and running. Assuming ElasticSearch is installed in home dir, running the service:

```shell
cd ~
./elasticsearch-8.10.4/bin/elasticsearch
```

Note that when launching ElasticSearch, you need to keep it running in the background (e.g., tmux or nohup).

Listing all indexes in ElasticSearch, port is 9200 by default:

```shell
curl -X GET "localhost:9200/_cat/indices?v"
```

Create index and run baseline (step=1 means single-step retrieval):

Before running the code, you need to uncomment the line `from elasticsearch import Elasticsearch` at the beginning of the script.

```shell
# MuSiQue
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset musique --corpus musique_1000
python src/baselines/retrieval_base.py --dataset musique --retriever bm25 --max_steps 1 --num_demo 0

# 2Wiki
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset 2wikimultihopqa --corpus 2wikimultihopqa_1000
python src/baselines/retrieval_base.py --dataset 2wikimultihopqa --retriever bm25 --max_steps 1 --num_demo 0

# HotpotQA
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset hotpotqa --corpus hotpotqa_1000
python src/baselines/retrieval_base.py --dataset hotpotqa --retriever bm25 --max_steps 1 --num_demo 0
```

### Contriever

Installing faiss and building faiss index is needed for Contriever.

```shell
export CUDA_VISIBLE_DEVICES=0

# MuSiQue
python src/baselines/mean_pooling_ip_faiss.py --dataset musique --model facebook/contriever
python src/baselines/retrieval_base.py --dataset musique --retriever facebook/contriever --max_steps 1 --num_demo 0

# 2Wiki
python src/baselines/mean_pooling_ip_faiss.py --dataset 2wikimultihopqa --model facebook/contriever
python src/baselines/retrieval_base.py --dataset 2wikimultihopqa --retriever facebook/contriever --max_steps 1 --num_demo 0

# HotpotQA
python src/baselines/mean_pooling_ip_faiss.py --dataset hotpotqa --model facebook/contriever
python src/baselines/retrieval_base.py --dataset hotpotqa --retriever facebook/contriever --max_steps 1 --num_demo 0

# NQ 
python src/baselines/mean_pooling_ip_faiss.py --dataset nq_rear --model facebook/contriever
python src/baselines/retrieval_base.py --dataset nq_rear --retriever facebook/contriever --max_steps 1 --num_demo 0

# popqa 
python src/baselines/mean_pooling_ip_faiss.py --dataset popqa --model facebook/contriever
python src/baselines/retrieval_base.py --dataset popqa --retriever facebook/contriever --max_steps 1 --num_demo 0

```

### BGE-M3
```shell
python src/baselines/create_index_bge.py --dataset musique --model BAAI/bge-m3
python src/baselines/retrieval_base.py --dataset musique --retriever BAAI/bge-m3 --max_steps 1 --num_demo 0

python src/baselines/create_index_bge.py --dataset 2wikimultihopqa --model BAAI/bge-m3
python src/baselines/retrieval_base.py --dataset 2wikimultihopqa --retriever BAAI/bge-m3 --max_steps 1 --num_demo 0

python src/baselines/create_index_bge.py --dataset hotpotqa --model BAAI/bge-m3
python src/baselines/retrieval_base.py --dataset hotpotqa --retriever BAAI/bge-m3 --max_steps 1 --num_demo 0
```






## Iter-RetGen
### Contriever
```shell
# musique
python src/baselines/mean_pooling_ip_faiss.py --dataset musique --model facebook/contriever
python src/baselines/iter-retgen.py --dataset musique --retriever facebook/contriever --max_steps 3 --top_k 5 --llm_model gpt-4o-mini

# 2wiki
python src/baselines/mean_pooling_ip_faiss.py --dataset 2wikimultihopqa --model facebook/contriever
python src/baselines/iter-retgen.py --dataset 2wikimultihopqa --retriever facebook/contriever --max_steps 3 --top_k 5 --llm_model gpt-4o-mini

# hotpot
python src/baselines/mean_pooling_ip_faiss.py --dataset hotpotqa --model facebook/contriever
python src/baselines/iter-retgen.py --dataset hotpotqa --retriever facebook/contriever --max_steps 3 --top_k 5 --llm_model gpt-4o-mini

# NQ
python src/baselines/mean_pooling_ip_faiss.py --dataset nq_rear --model facebook/contriever
python src/baselines/iter-retgen.py --dataset nq_rear --retriever facebook/contriever --max_steps 3 --top_k 5 --llm_model gpt-4o-mini

# PopQA
python src/baselines/mean_pooling_ip_faiss.py --dataset popqa --model facebook/contriever
python src/baselines/iter-retgen.py --dataset popqa --retriever facebook/contriever --max_steps 3 --top_k 5 --llm_model gpt-4o-mini
```


### BGE-M3

```shell
# MuSiQue
python src/baselines/create_index_bge.py --dataset musique --model BAAI/bge-m3 --dim 1024
python src/baselines/iter-retgen.py --dataset musique --retriever BAAI/bge-m3 --max_steps 3 --top_k 5 --llm_model gpt-4o-mini
# 2Wiki
python src/baselines/create_index_bge.py --dataset 2wikimultihopqa --model BAAI/bge-m3 --dim 1024
python src/baselines/iter-retgen.py --dataset 2wikimultihopqa --retriever BAAI/bge-m3 --max_steps 3 --top_k 5 --llm_model gpt-4o-mini
# hotpot
python src/baselines/create_index_bge.py --dataset hotpotqa --model BAAI/bge-m3 --dim 1024
python src/baselines/iter-retgen.py --dataset hotpotqa --retriever BAAI/bge-m3 --max_steps 3 --top_k 5 --llm_model gpt-4o-mini
```







## IRCoT

If index has been created during single-step retrieval, you could skip the indexing step and call `ircot.py` directly.
To run BM25, set up ElasticSearch as described above.

### IRCoT BM25

```shell
# MuSiQue
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset musique --corpus musique_1000
python src/baselines/ircot.py --dataset musique --retriever bm25 --max_steps 4 --num_demo 1

# 2Wiki
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset 2wikimultihopqa --corpus 2wikimultihopqa_1000
python src/baselines/ircot.py --dataset 2wikimultihopqa --retriever bm25 --max_steps 2 --num_demo 1

# HotpotQA
python src/baselines/create_retrieval_index.py --retriever bm25 --dataset hotpotqa --corpus hotpotqa_1000
python src/baselines/ircot.py --dataset hotpotqa --retriever bm25 --max_steps 2 --num_demo 1
```

### IRCoT Contriever

```shell
# MuSiQue
python src/baselines/mean_pooling_ip_faiss.py --dataset musique --model facebook/contriever
python src/baselines/ircot_246.py --dataset musique --retriever facebook/contriever --max_steps 3 --num_demo 1 --llm_model gpt-4o-mini

# 2Wiki
python src/baselines/mean_pooling_ip_faiss.py --dataset 2wikimultihopqa --model facebook/contriever
python src/baselines/ircot_246.py --dataset 2wikimultihopqa --retriever facebook/contriever --max_steps 3 --num_demo 1

# HotpotQA
python src/baselines/mean_pooling_ip_faiss.py --dataset hotpotqa --model facebook/contriever
python src/baselines/ircot_246.py --dataset hotpotqa --retriever facebook/contriever --max_steps 3 --num_demo 1 --llm_model gpt-4o-mini

# NQ
python src/baselines/mean_pooling_ip_faiss.py --dataset nq_rear --model facebook/contriever
python src/baselines/ircot_246.py --dataset nq_rear --retriever facebook/contriever --max_steps 3 --num_demo 1 --llm_model gpt-4o-mini

# popqa
python src/baselines/mean_pooling_ip_faiss.py --dataset popqa --model facebook/contriever
python src/baselines/ircot_246.py --dataset popqa --retriever facebook/contriever --max_steps 3 --num_demo 1 --llm_model gpt-4o-mini
```


### IRCoT bge-m3

```shell
# MuSiQue
python src/baselines/create_index_bge.py --dataset musique --model BAAI/bge-m3
python src/baselines/ircot_246.py --dataset musique --retriever BAAI/bge-m3 --max_steps 3 --num_demo 1 --llm_model gpt-4o-mini

# 2Wiki
python src/baselines/create_index_bge.py --dataset 2wikimultihopqa --model BAAI/bge-m3
python src/baselines/ircot_246.py --dataset 2wikimultihopqa --retriever BAAI/bge-m3 --max_steps 3 --num_demo 1 --llm_model gpt-4o-mini

# HotpotQA
python src/baselines/create_index_bge.py --dataset hotpotqa --model BAAI/bge-m3
python src/baselines/ircot_246.py --dataset hotpotqa --retriever BAAI/bge-m3 --max_steps 3 --num_demo 1 --llm_model gpt-4o-mini
```


