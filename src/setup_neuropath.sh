data=$1  # e.g., 'sample'
retriever_name=$2  # e.g., 'facebook/contriever'
index_llm_model=$3 # e.g., 'gpt-4o-mini' (OpenAI)
llm_api=$4 # e.g., 'openai', 'together'
extraction_type=ner

# Running Open Information Extraction
python src/openie_with_retrieval_option_parallel.py --dataset $data --llm $llm_api --model_name $index_llm_model --run_ner --num_passages all # NER and OpenIE for passages
python src/query_ner_vtp_parallel.py --dataset $data --llm $llm_api --model_name $index_llm_model  # NER for queries

# Building the graph
python src/create_graph.py --dataset $data --model_name $retriever_name --index_llm_model $index_llm_model --extraction_type $extraction_type
python src/create_graph.py --dataset $data --model_name $retriever_name --index_llm_model $index_llm_model --create_graph --extraction_type $extraction_type
