Use `qa_reader.py` to leverage retrieved documents to answer the questions, e.g.,

```shell
# example
python src/qa/qa_reader.py --dataset hotpotqa --prefix naive --retriever bm25 --data output/base_retriever/retrieved_hotpotqa_bm25_.json

python src/qa/qa_reader.py --dataset hotpotqa --prefix neuropath --retriever contriever --data output/retrieved/retrieved_results_hotpotqa_neuropath_facebook_contriever_gpt-4o-mini_top_100.json
```

The "retriever" argument is only used for naming purposes, while the "data" argument specifies the JSON file where the retrieval results are saved.