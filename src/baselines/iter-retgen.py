import argparse
import os
import sys
sys.path.append('.')
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
from data.iter_retgen_prompts import prompts
import faiss
from threading import Lock
from src.processing import extract_json_dict
import numpy as np
from sentence_transformers import SentenceTransformer
# for large embedding models
from src.lm_wrapper.util import init_embedding_model

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))




from abc import abstractmethod
import torch
from transformers import AutoTokenizer, AutoModel
from src.processing import mean_pooling_embedding_with_normalization
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from src.langchain_util import init_langchain_model, num_tokens_by_tiktoken

class DocumentRetriever:
    @abstractmethod
    def rank_docs(self, query: str, top_k: int):
        """
        Rank the documents in the corpus based on the given query
        :param query:
        :param top_k:
        :return: ranks and scores of the retrieved documents
        """

class DPRRetriever(DocumentRetriever):
    def __init__(self, model_name: str, faiss_index: str, corpus, device='cuda'):
        """

        :param model_name:
        :param faiss_index: The path to the faiss index
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        self.faiss_index = faiss_index
        self.corpus = corpus
        self.device = device

    def rank_docs(self, query: str, top_k: int):
        # query_embedding = mean_pooling_embedding(query, self.tokenizer, self.model, self.device)
        with torch.no_grad():
            query_embedding = mean_pooling_embedding_with_normalization(query, self.tokenizer, self.model, self.device).detach().cpu().numpy()
        inner_product, corpus_idx = self.faiss_index.search(query_embedding, top_k)

        return corpus_idx.tolist()[0], inner_product.tolist()[0]

class SentenceTransformersRetriever(DocumentRetriever):
    def __init__(self, model_name: str, faiss_index: str, corpus, device='cuda', norm=True):
        """

        :param model_name:
        :param faiss_index: The path to the faiss index
        """
        self.model = SentenceTransformer(model_name)
        self.faiss_index = faiss_index
        self.corpus = corpus
        self.device = device
        self.norm = norm

    def rank_docs(self, query: str, top_k: int):
        query_embedding = self.model.encode(query)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        if self.norm:
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_embedding = query_embedding / norm
        inner_product, corpus_idx = self.faiss_index.search(query_embedding, top_k)

        return corpus_idx.tolist()[0], inner_product.tolist()[0]

# large embedding model
class LMRetriever(DocumentRetriever):
    def __init__(self, model_name: str, faiss_index: str, corpus, device='cuda', norm=True):
        """
        
        :param model_name:
        :param faiss_index: The path to the faiss index
        """
        self.model = init_embedding_model(model_name)
        self.faiss_index = faiss_index
        self.corpus = corpus
        self.device = device
        self.norm = norm

    def rank_docs(self, query: str, top_k: int):
        try:
            query_embedding = self.model.encode(query, 
                                                instruction = 'Given a question, retrieve relevant documents that best answer the question.'
                                               )
            
        except Exception as e:
            print("[encode error]", e)
        # query_embedding = np.expand_dims(query_embedding, axis=0)
        if self.norm:
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_embedding = query_embedding / norm

        try:
            query_embedding = query_embedding.astype(np.float32) 
            inner_product, corpus_idx = self.faiss_index.search(query_embedding, top_k)
        except Exception as e:
            print("[faiss search error]", e)
        # inner_product, corpus_idx = self.faiss_index.search(query_embedding, top_k)
        return corpus_idx.tolist()[0], inner_product.tolist()[0]



def retrieve_step(query: str, corpus, top_k: int, retriever: DocumentRetriever, dataset: str):
    doc_ids, scores = retriever.rank_docs(query, top_k=top_k)
    if dataset in ['hotpotqa']:
        retrieved_passages = []
        for doc_id in doc_ids:
            key = list(corpus.keys())[doc_id]
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    elif dataset in ['musique', '2wikimultihopqa', 'nq_rear', 'popqa', 'narrativeqa_dev_10_doc']:
        retrieved_passages = [corpus[doc_id]['title'] + '\n' + corpus[doc_id]['text'] for doc_id in doc_ids]
    elif dataset in ['multihoprag', 'multihoprag_chunks']:
        retrieved_passages = [corpus[doc_id]['text'] for doc_id in doc_ids]
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')
    return retrieved_passages, scores


class IterRetGen:
    def __init__(
        self,
        args,
        client: str,
        data: dict,
        corpus,
        retriever: DocumentRetriever,
        max_iter: int = 3,
        topk: int = 10, 
        sys_promt: str = prompts.SYSTEM_PROMPT,
        prompt_template: str = None,
        processed_ids: set = None,
        k_list: list = None
    ) -> None:
        self.args = args
        self.client = client
        self.data = data
        self.corpus = corpus
        self.retriever = retriever
        self.max_iter = max_iter
        self.topk = topk
        self.sys_promt = sys_promt
        self.prompt_template = prompt_template
        self.k_list = k_list
        self.processed_ids = processed_ids

    def inference(self, workers=10, results:dict=None) -> list[dict]:
        print("Start inference")
        lock = Lock()  # ensure atomic state updates across threads
        total_processing_time = 0
        valid_sample_count = 0
        total_tokens = 0
        total_recall = {k: 0 for k in self.k_list}
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self.process_sample, idx, sample, self.args, self.corpus, self.retriever, self.client, self.processed_ids)
                for idx, sample in enumerate(self.data)
            ]
            
            for future in tqdm(as_completed(futures), total=len(data), desc='Parallel Iter-RetGen'):
                idx, recall, retrieved_passages, thoughts, tokens, elapsed_time = future.result()
                with lock:
                    total_processing_time += elapsed_time
                    total_tokens += tokens
                    valid_sample_count += 1
                    # print metrics
                    for k in k_list:
                        total_recall[k] += recall[k]
                        print(f'R@{k}: {total_recall[k] / (idx + 1):.4f} ', end='')
                    print()
                print('[THOUGHT]', thoughts)
                
                # record results
                results[idx]['retrieved'] = retrieved_passages
                results[idx]['recall'] = recall
                results[idx]['thoughts'] = thoughts
                
                if idx % 50 == 0:
                    with open(output_path, 'w') as f:
                        json.dump(results, f)
            print(f"Total processing time: {total_processing_time:.4f} seconds")
            print(f"Average processing time per sample: {total_processing_time/valid_sample_count:.4f} seconds")
            print(f"Total tokens: {total_tokens} tokens")
            

            

    def construct_prompt(self, sample: dict, prev_answer: str = None) -> str:
        q = sample["question"]
        if prev_answer is not None:
            q = f"{q} {prev_answer}"

        retrieved_passages, scores = retrieve_step(q, corpus, args.top_k, retriever, args.dataset)
        
        
        docs = self.retriever.search(q, top_k=self.topk)
        d = "\n".join([f"{doc['text']}" for doc in docs[0]])
        
        
        d = "\n".join([f"{doc}" for doc in retrieved_passages])
        
        prompt = self.prompt_template.format(documents=d, question=sample["question"])
        return prompt, docs

    def process_sample(self, idx, sample, args, corpus, retriever, client, processed_ids):
        start_time = time.time()
        # Check if the sample has already been processed
        if args.dataset in ['hotpotqa', '2wikimultihopqa']:
            sample_id = sample['_id']
        elif args.dataset in ['musique', 'nq_rear', 'popqa', 'multihoprag', 'multihoprag_chunks']:
            sample_id = sample['id']
        elif args.dataset in ['narrativeqa_dev_10_doc']:
            sample_id = None
        else:
            raise NotImplementedError(f'Dataset {args.dataset} not implemented')
        
        if sample_id in processed_ids:
            return  # Skip already processed samples
        
        
        # Perform retrieval and reasoning steps
        try:
            if sample.get('question', None):
                query = sample['question'] 
            else:
                query = sample['query'] 
        except Exception as e:
            print(e)
        try:
            docs, scores = retrieve_step(query, corpus, self.topk, self.retriever, args.dataset)
        except Exception as e:
            print("[retrieve_step error]: ", e)
        
        thoughts = []
        
        cur_iter = 1
        total_tokens = 0

        # previous answer starts empty
        prev_answer = ""
        while cur_iter < self.max_iter:
            cur_iter += 1
            q = sample["question"]
            
            d = "\n".join([f"{doc}" for doc in docs])
            prompt = self.prompt_template.format(documents=d, question=q)
            
            check_if_valid = (
                lambda x: type(x) is dict and "thought" in x
                # lambda x: type(x) is dict and "answer" in x and "thought" in x
            )
            try:
                chat_completion = client.invoke(
                    [
                        SystemMessage(sys_promt),
                        HumanMessage(prompt)
                    ]
                )
                model_response = chat_completion.content
                model_response = extract_json_dict(model_response)
                tokens = chat_completion.response_metadata['token_usage']['total_tokens']
                total_tokens += tokens
                if check_if_valid is not None and not check_if_valid(model_response):
                    print(f"Invalid response {model_response}")
                    model_response = {}
                    model_response["thought"] = ""
                    # model_response["answer"] = ""
            except Exception as e:
                print("Response error, ", e)
                model_response = {}
                model_response["thought"] = ""

            prev_answer = model_response["thought"]
            thoughts.append(prev_answer)
            
            if prev_answer != "":
                q = f"{q} {prev_answer}"

            new_docs, new_scores = retrieve_step(q, corpus, self.topk, self.retriever, args.dataset)
            docs = new_docs
            
        # calculate recall
        try:
            if args.dataset in ['hotpotqa']:
                gold_passages = [item for item in sample['supporting_facts']]
                gold_items = set([item[0] for item in gold_passages])
                retrieved_items = [passage.split('\n')[0].strip() for passage in docs]
            elif args.dataset in ['musique']:
                gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
                gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
                retrieved_items = docs
            elif args.dataset in ['2wikimultihopqa']:
                gold_passages = [item for item in sample['supporting_facts']]
                gold_items = set([item[0] for item in gold_passages])
                retrieved_items = [passage.split('\n')[0].strip() for passage in docs]
            elif args.dataset in ['nq_rear']:
                gold_passages = [item for item in sample['contexts'] if item['is_supporting']]
                gold_items = set([item['title'] + '\n' + item['text'] for item in gold_passages])
                retrieved_items = docs
            elif args.dataset in ['popqa']:
                gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
                gold_items = set([item['title'] + '\n' + item['text'] for item in gold_passages])
                retrieved_items = docs
            # multihop-rag
            elif args.dataset in ['multihoprag', 'multihoprag_chunks']:
                gold_passages = [item for item in sample['evidence_list']]
                gold_items = set([item['fact'] for item in gold_passages])
                retrieved_items = docs
            # narrativeqa_dev_10_doc
            elif args.dataset in ['narrativeqa_dev_10_doc']:
                gold_passages = [' ']
                gold_items = [' ']
                retrieved_items = docs
            else:
                raise NotImplementedError(f'Dataset {args.dataset} not implemented')
            recall = dict()
            print(f'idx: {idx + 1} ', end='')
            for k in k_list:
                if args.dataset in ['multihoprag', 'multihoprag_chunks']:
                    recall[k] = round(
                                    sum(
                                        1 for gold in gold_items
                                        if any(gold in retrieved for retrieved in retrieved_items[:k])
                                    ) / len(gold_items), 4
                                )
                else:
                    recall[k] = round(sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items), 4)
            
            elapsed_time = time.time() - start_time 
        except Exception as e:
                print("[calculate recall error, ", e)
        return idx, recall, docs, thoughts, total_tokens, elapsed_time

        
    
    def infer_sample(self, sample: dict) -> dict:
        question = sample["question"]
        results = {
            "id": sample["id"],
            "answer": sample["answer"],
            "oracle": [
                f"{sample['id']}-{'{:02d}'.format(chunk['positive_paragraph_idx'])}"
                for chunk in sample["decomposed_questions"].values()
            ],
            "question": question,
            "inter": {},
        }
        internal_query = None
        cur_iter = 0
        while cur_iter < self.max_iter:
            cur_iter += 1
            prompt, docs = self.construct_prompt(sample, internal_query)
            check_if_valid = (
                lambda x: type(x) is dict and "answer" in x and "thought" in x
            )
            model_response = ask_model(
                self.model,
                prompt,
                system_msg=SYSTEM_PROMPT,
                type="json",
                mode="chat",
                check_if_valid=check_if_valid,  # noqa
            )
            internal_query = model_response["thought"]
            results["inter"][cur_iter] = {
                "cur_answer": model_response["answer"],
                "thought": model_response["thought"],
                "docs": docs,
            }
        results["model_answer"] = model_response["answer"]
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['hotpotqa', 'musique', '2wikimultihopqa', 'nq_rear', 'popqa', 'multihoprag', 'multihoprag_chunks', 'narrativeqa_dev_10_doc'], required=True)
    parser.add_argument('--llm', type=str, default='openai', help="LLM provider (only 'openai' is supported; use an OpenAI-compatible base_url for vLLM/Ollama).")
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--top_k', type=int, default=10, help='retrieving k documents at each step')
    parser.add_argument('--thread', type=int, default=10, help='number of threads for parallel processing, 1 for sequential processing')
    args = parser.parse_args()
    
    
    
    start = time.time()
    
    retriever_name = args.retriever.replace('/', '_').replace('.', '_')
    client = init_langchain_model(args.llm, args.llm_model)
    # load dataset and corpus
    sys_promt = prompts.SYSTEM_PROMPT
    if args.dataset == 'hotpotqa':
        data = json.load(open('data/hotpotqa.json', 'r'))
        corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
        prompt_template = prompts.ITER_RETGEN_HOTPOTQA_PROMPT
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'musique':
        data = json.load(open('data/musique.json', 'r'))
        corpus = json.load(open('data/musique_corpus.json', 'r'))
        prompt_template = prompts.ITER_RETGEN_MUSIQUE_PROMPT
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == '2wikimultihopqa':
        data = json.load(open('data/2wikimultihopqa.json', 'r'))
        corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        prompt_template = prompts.ITER_RETGEN_WIKIMQA_PROMPT
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'nq_rear':
        data = json.load(open('data/nq_rear.json', 'r'))
        corpus = json.load(open('data/nq_rear_corpus.json', 'r'))
        prompt_template = prompts.ITER_RETGEN_WIKIMQA_PROMPT  # reuse the prompt
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'popqa':
        data = json.load(open('data/popqa.json', 'r'))
        corpus = json.load(open('data/popqa_corpus.json', 'r'))
        prompt_template = prompts.ITER_RETGEN_WIKIMQA_PROMPT  # reuse the prompt
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'multihoprag' or args.dataset == 'multihoprag_chunks':
        data = json.load(open(f'data/{args.dataset}.json', 'r'))
        corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
        prompt_template = prompts.ITER_RETGEN_WIKIMQA_PROMPT  # reuse the prompt
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'narrativeqa_dev_10_doc':
        data = json.load(open(f'data/{args.dataset}.json', 'r'))
        corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
        prompt_template = prompts.ITER_RETGEN_WIKIMQA_PROMPT  # reuse the prompt
        max_steps = args.max_steps if args.max_steps is not None else 2
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')
    
    output_path = f"output/iter_retgen/{args.dataset}_{retriever_name}_{args.llm_model}_step_{max_steps}_top_{args.top_k}.json"
    

    if args.retriever == 'facebook/contriever':
        if args.dataset == 'hotpotqa':
            faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_facebook_contriever_ip_norm.index')
        elif args.dataset == 'musique':
            faiss_index = faiss.read_index('data/musique/musique_facebook_contriever_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_facebook_contriever_ip_norm.index')
        elif args.dataset == 'nq_rear':
            faiss_index = faiss.read_index('data/nq_rear/nq_rear_facebook_contriever_ip_norm.index')
        elif args.dataset == 'popqa':
            faiss_index = faiss.read_index('data/popqa/popqa_facebook_contriever_ip_norm.index')
        elif args.dataset == 'multihoprag' or args.dataset == 'multihoprag_chunks':
            faiss_index = faiss.read_index(f'data/{args.dataset}/{args.dataset}_facebook_contriever_ip_norm.index')
        retriever = DPRRetriever(args.retriever, faiss_index, corpus)

    elif args.retriever == 'BAAI/bge-m3':  # 
        if args.dataset == 'hotpotqa':
            faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_BAAI_bge-m3_ip_norm.index')
        elif args.dataset == 'musique':
            faiss_index = faiss.read_index('data/musique/musique_BAAI_bge-m3_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_BAAI_bge-m3_ip_norm.index')
        elif args.dataset == 'nq_rear':
            faiss_index = faiss.read_index('data/nq_rear/nq_rear_BAAI_bge-m3_ip_norm.index')
        elif args.dataset == 'popqa':
            faiss_index = faiss.read_index('data/popqa/popqa_BAAI_bge-m3_ip_norm.index')
        elif args.dataset == 'multihoprag' or args.dataset == 'multihoprag_chunks':
            faiss_index = faiss.read_index(f'data/{args.dataset}/{args.dataset}_BAAI_bge-m3_ip_norm.index')
        retriever = SentenceTransformersRetriever(args.retriever, faiss_index, corpus)
    # large embedding model
    elif args.retriever.startswith('Alibaba-NLP/gte-Qwen2-7B-instruct'):
        if args.dataset == 'hotpotqa':
            faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_Alibaba-NLP_gte-Qwen2-7B-instruct_ip_norm.index')
        elif args.dataset == 'musique':
            faiss_index = faiss.read_index('data/musique/musique_Alibaba-NLP_gte-Qwen2-7B-instruct_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_Alibaba-NLP_gte-Qwen2-7B-instruct_ip_norm.index')
        retriever = LMRetriever(args.retriever, faiss_index, corpus)
    elif args.retriever.startswith('nvidia/NV-Embed-v2'):
        if args.dataset == 'hotpotqa':
            faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_nvidia_NV-Embed-v2_ip_norm.index')
        elif args.dataset == 'musique':
            faiss_index = faiss.read_index('data/musique/musique_nvidia_NV-Embed-v2_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_nvidia_NV-Embed-v2_ip_norm.index')
        elif args.dataset == 'narrativeqa_dev_10_doc':
            faiss_index = faiss.read_index('data/narrativeqa_dev_10_doc/narrativeqa_dev_10_doc_nvidia_NV-Embed-v2_ip_norm.index')
        retriever = LMRetriever(args.retriever, faiss_index, corpus)
    
    
    k_list = [1, 2, 5, 10, 15, 20, 30, 50, 100]
    total_recall = {k: 0 for k in k_list}
    
    # read previous results
    results = data
    read_existing_data = False
    try:
        if os.path.isfile(output_path):
            with open(output_path, 'r') as f:
                results = json.load(f)
                print(f'Loaded {len(results)} results from {output_path}')
                if len(results):
                    read_existing_data = True
        if args.dataset in ['hotpotqa', '2wikimultihopqa']:
            processed_ids = {sample['_id'] for sample in results if 'retrieved' in sample}
        elif args.dataset in ['musique', 'nq_rear', 'popqa', 'multihoprag', 'multihoprag_chunks']:
            processed_ids = {sample['id'] for sample in results if 'retrieved' in sample}
        elif args.dataset in ['narrativeqa_dev_10_doc']:
            processed_ids = {}
        else:
            raise NotImplementedError(f'Dataset {args.dataset} not implemented')
        for sample in results:
            if 'recall' in sample:
                total_recall = {k: total_recall[k] + sample['recall'][str(k)] for k in k_list}
    except Exception as e:
        print('loading results exception', e)
        print(f'Results file {output_path} maybe empty, cannot be loaded.')
        processed_ids = set()
    
    
    # data = data[:2]
    retgen = IterRetGen(
        args, client, data, corpus, retriever, max_iter=max_steps, topk=args.top_k,
        sys_promt = sys_promt,
        prompt_template=prompt_template,
        k_list = k_list,
        processed_ids=processed_ids
    )
    
    
    
    
    
    if len(results) > 0:
        for k in k_list:
            print(f'R@{k}: {total_recall[k] / len(results):.4f} ', end='')
        print()
    
    if read_existing_data:
        print(f'All samples have been already in the result file ({output_path}), exit.')
        exit(0)
    
    
    retgen.inference(workers=args.thread, results=results)
    
    # save results
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f'Saved results to {output_path}')
    
    end = time.time()
    print(f"Total time: {end - start:.2f}s")


    