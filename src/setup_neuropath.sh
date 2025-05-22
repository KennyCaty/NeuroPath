data=$1  # e.g., 'sample'
retriever_name=$2  # e.g., 'facebook/contriever'
extraction_model=$3 # e.g., 'gpt-4o-mini' (OpenAI)
available_gpus=$4
syn_thresh=$5 # float, e.g., 0.8
llm_api=$6 # e.g., 'openai', 'together'
extraction_type=ner

# Running Open Information Extraction
python src/openie_with_retrieval_option_parallel.py --dataset $data --llm $llm_api --model_name $extraction_model --run_ner --num_passages all # NER and OpenIE for passages
python src/query_ner_vtp_parallel.py --dataset $data --llm $llm_api --model_name $extraction_model  # NER for queries

# Creating Contriever Graph
python src/create_graph.py --dataset $data --model_name $retriever_name --extraction_model $extraction_model --threshold $syn_thresh --extraction_type $extraction_type

CUDA_VISIBLE_DEVICES=$available_gpus python src/RetrievalModule.py --retriever_name $retriever_name --string_filename output/query_to_kb.tsv
CUDA_VISIBLE_DEVICES=$available_gpus python src/RetrievalModule.py --retriever_name $retriever_name --string_filename output/kb_to_kb.tsv
CUDA_VISIBLE_DEVICES=$available_gpus python src/RetrievalModule.py --retriever_name $retriever_name --string_filename output/rel_kb_to_kb.tsv

# Remove --cosine_sim_edges, as generating similarity edges is not required.
python src/create_graph.py --dataset $data --model_name $retriever_name --extraction_model $extraction_model --threshold $syn_thresh --create_graph --extraction_type $extraction_type 
