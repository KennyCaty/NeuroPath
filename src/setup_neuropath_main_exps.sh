gpus_available=0
syn_threshold=0.8

# The loop here is provided only for illustrative purposes. For actual runs, we recommend running the command separately for each dataset. For example: for data in 2wikimultihopqa;
# for data in musique 2wikimultihopqa hotpotqa popqa nq_rear;
# do
#   # bash src/setup_neuropath.sh $data BAAI/bge-m3 gpt-4o-mini $gpus_available $syn_threshold openai
#   bash src/setup_neuropath.sh $data facebook/contriever gpt-4o-mini $gpus_available $syn_threshold openai
# done

# ====== Example ======
data=2wikimultihopqa
bash src/setup_neuropath.sh $data facebook/contriever gpt-4o-mini $gpus_available $syn_threshold openai

