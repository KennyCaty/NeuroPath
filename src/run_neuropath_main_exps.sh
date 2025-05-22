## ==== All Datasets ===
# musique 2wikimultihopqa hotpotqa popqa nq_rear

## ====Run Example ===
python3 src/rag_neuropath.py --dataset 2wikimultihopqa --retriever facebook/contriever --top_k 100  --llm_model gpt-4o-mini  --one_shot f --max_hop 2
