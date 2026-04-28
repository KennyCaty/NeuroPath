import sys

sys.path.append('.')

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_util import init_langchain_model, num_tokens_by_tiktoken
from src.processing import mean_pooling_embedding_with_normalization
from src.elastic_search_tool import search_with_score
import numpy as np
from sentence_transformers import SentenceTransformer
# for large embedding models
from src.lm_wrapper.util import init_embedding_model

import torch
import concurrent
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor

import faiss
# from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel

import argparse
import json
import os

from tqdm import tqdm

ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'


class DocumentRetriever:
    @abstractmethod
    def rank_docs(self, query: str, top_k: int):
        """
        Rank the documents in the corpus based on the given query
        :param query:
        :param top_k:
        :return: ranks and scores of the retrieved documents
        """


class BM25Retriever(DocumentRetriever):
    def __init__(self, index_name: str, host: str = 'localhost', port: int = 9200):
        self.es = Elasticsearch([{"host": host, "port": port, "scheme": "http"}], max_retries=5, retry_on_timeout=True, request_timeout=30)
        self.index_name = index_name

    def rank_docs(self, query: str, top_k: int):
        results = search_with_score(self.es, self.index_name, query, top_k)
        return [int(result[0]) for result in results], [result[1] for result in results]


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


def parse_prompt(file_path: str, has_context=True):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content by the metadata pattern
    parts = content.split('# METADATA: ')
    parsed_data = []
    if has_context:
        for part in parts[1:]:  # Skip the first split as it will be empty
            metadata_section, rest_of_data = part.split('\n', 1)
            metadata = json.loads(metadata_section)
            document_sections = rest_of_data.strip().split('\n\nQ: ')
            document_text = document_sections[0].strip()
            qa_pair = document_sections[1].split('\nA: ')
            question = qa_pair[0].strip()
            thought_and_answer = qa_pair[1].strip().split('So the answer is: ')
            thought = thought_and_answer[0].strip()
            answer = thought_and_answer[1].strip()

            parsed_data.append({
                'metadata': metadata,
                'document': document_text,
                'question': question,
                'thought_and_answer': qa_pair[1].strip(),
                'thought': thought,
                'answer': answer
            })
    else:
        for part in parts[1:]:
            metadata_section, rest_of_data = part.split('\n', 1)
            metadata = json.loads(metadata_section)
            s = rest_of_data.split('\n')
            question = s[0][3:].strip()
            thought_and_answer = s[1][3:].strip().split('So the answer is: ')
            thought = thought_and_answer[0].strip()
            answer = thought_and_answer[1].strip()

            parsed_data.append({
                'metadata': metadata,
                'question': question,
                'thought_and_answer': s[1][3:].strip(),
                'thought': thought,
                'answer': answer
            })

    return parsed_data


def retrieve_step(query: str, corpus, top_k: int, retriever: DocumentRetriever, dataset: str):
    try:
        doc_ids, scores = retriever.rank_docs(query, top_k=top_k)
    except Exception as e:
        print("[rank docs error]", e)
    if dataset in ['hotpotqa']:
        retrieved_passages = []
        for doc_id in doc_ids:
            key = list(corpus.keys())[doc_id]
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    elif dataset in ['musique', '2wikimultihopqa', 'nq_rear', 'popqa']:
        retrieved_passages = [corpus[doc_id]['title'] + '\n' + corpus[doc_id]['text'] for doc_id in doc_ids]
    elif dataset in ['multihoprag', 'multihoprag_chunks']:
        retrieved_passages = [corpus[doc_id]['text'] for doc_id in doc_ids]
    else:
        raise NotImplementedError(f'Dataset {dataset} not implemented')
    return retrieved_passages, scores


def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
    merged_dict = {}

    # Iterate through each element in the list
    for element in elements:
        # Split the element into lines and get the first line
        lines = element.split('\n')
        first_line = lines[0]

        # Check if the first line is already a key in the dictionary
        if first_line in merged_dict:
            # Append the current element to the existing value
            merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
        else:
            # Add the current element as a new entry in the dictionary
            merged_dict[first_line] = prefix + element

    # Extract the merged elements from the dictionary
    merged_elements = list(merged_dict.values())
    return merged_elements


def reason_step(dataset, few_shot: list, query: str, passages: list, thoughts: list, client):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with LangChain LLM.
    The generated thought is used for further retrieval step.
    :return: next thought
    """
    prompt_demo = ''

    prompt_user = ''
    if dataset in ['hotpotqa']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'Wikipedia Title: {passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    for sample in few_shot:
        cur_sample = f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["thought_and_answer"]}\n\n'
        if num_tokens_by_tiktoken(ircot_reason_instruction + prompt_demo + cur_sample + prompt_user) < 15000:
            prompt_demo += cur_sample

    messages = ChatPromptTemplate.from_messages([SystemMessage(ircot_reason_instruction + '\n\n' + prompt_demo),
                                                 HumanMessage(prompt_user)]).format_prompt()

    try:
        chat_completion = client.invoke(messages.to_messages())
        response_content = chat_completion.content
    except Exception as e:
        print(e)
        return ''
    return response_content


def process_sample(idx, sample, args, corpus, retriever, client, processed_ids):
    # Check if the sample has already been processed
    if args.dataset in ['hotpotqa', '2wikimultihopqa']:
        sample_id = sample['_id']
    elif args.dataset in ['musique']:
        sample_id = sample['id']
    elif args.dataset in ['nq_rear', 'popqa', 'multihoprag', 'multihoprag_chunks']:
        sample_id = sample['id']
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

    if sample_id in processed_ids:
        return  # Skip already processed samples

    # Perform retrieval and reasoning steps
    if sample.get('question', None):
        query = sample['question'] 
    else:
        query = sample['query'] 
    try:
        retrieved_passages, scores = retrieve_step(query, corpus, args.top_k, retriever, args.dataset)
    except Exception as e:
        print('[error]: ', e)
        
    thoughts = []
    retrieved_passages_dict = {passage: score for passage, score in zip(retrieved_passages, scores)}
    it = 1
    for it in range(1, max_steps):
        new_thought = reason_step(args.dataset, few_shot_samples, query, retrieved_passages[:args.top_k], thoughts, client)
        thoughts.append(new_thought)
        if 'So the answer is:' in new_thought:
            break
        new_retrieved_passages, new_scores = retrieve_step(new_thought, corpus, args.top_k, retriever, args.dataset)

        for passage, score in zip(new_retrieved_passages, new_scores):
            if passage in retrieved_passages_dict:
                retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
            else:
                retrieved_passages_dict[passage] = score

        retrieved_passages, scores = zip(*retrieved_passages_dict.items())

        sorted_passages_scores = sorted(zip(retrieved_passages, scores), key=lambda x: x[1], reverse=True)
        retrieved_passages, scores = zip(*sorted_passages_scores)
    # end iteration

    # calculate recall
    if args.dataset in ['hotpotqa']:
        gold_passages = [item for item in sample['supporting_facts']]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    elif args.dataset in ['musique']:
        gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
        gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
        retrieved_items = retrieved_passages
    elif args.dataset in ['2wikimultihopqa']:
        gold_passages = [item for item in sample['supporting_facts']]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    elif args.dataset in ['nq_rear']:
        gold_passages = [item for item in sample['contexts'] if item['is_supporting']]
        gold_items = set([item['title'] + '\n' + item['text'] for item in gold_passages])
        retrieved_items = retrieved_passages
    elif args.dataset in ['popqa']:
        gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
        gold_items = set([item['title'] + '\n' + item['text'] for item in gold_passages])
        retrieved_items = retrieved_passages
    # multihop-rag
    elif args.dataset in ['multihoprag', 'multihoprag_chunks']:
        gold_passages = [item for item in sample['evidence_list']]
        gold_items = set([item['fact'] for item in gold_passages])
        retrieved_items = retrieved_passages
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

    recall = dict()
    print(f'idx: {idx + 1} ', end='')
    # for k in k_list:
    #     recall[k] = sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items)
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
    return idx, recall, retrieved_passages, thoughts, it


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['hotpotqa', 'musique', '2wikimultihopqa', 'nq_rear', 'popqa', 'multihoprag', 'multihoprag_chunks'], required=True)
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    parser.add_argument('--num_demo', type=int, default=1, help='the number of documents in the demonstration', required=True)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--top_k', type=int, default=100, help='retrieving k documents at each step')
    parser.add_argument('--thread', type=int, default=6, help='number of threads for parallel processing, 1 for sequential processing')
    args = parser.parse_args()

    retriever_name = args.retriever.replace('/', '_').replace('.', '_')
    client = init_langchain_model(args.llm, args.llm_model)

    # load dataset and corpus
    if args.dataset == 'hotpotqa':
        data = json.load(open('data/hotpotqa.json', 'r'))
        corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'musique':
        data = json.load(open('data/musique.json', 'r'))
        corpus = json.load(open('data/musique_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = args.max_steps if args.max_steps is not None else 4
    elif args.dataset == '2wikimultihopqa':
        data = json.load(open('data/2wikimultihopqa.json', 'r'))
        corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'nq_rear':
        data = json.load(open('data/nq_rear.json', 'r'))
        corpus = json.load(open('data/nq_rear_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'popqa':
        data = json.load(open('data/popqa.json', 'r'))
        corpus = json.load(open('data/popqa_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'multihoprag' or args.dataset == 'multihoprag_chunks':
        data = json.load(open(f'data/{args.dataset}.json', 'r'))
        corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
        prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        max_steps = args.max_steps if args.max_steps is not None else 2
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')
    few_shot_samples = parse_prompt(prompt_path)
    few_shot_samples = few_shot_samples[:args.num_demo]
    print('num of demo:', len(few_shot_samples))

    if max_steps > 1:
        output_path = f'output/ircot/{args.dataset}_{retriever_name}_demo_{args.num_demo}_{args.llm_model}_step_{max_steps}_top_{args.top_k}.json'
    else:  # only one step
        args.top_k = 100
        output_path = f'output/base_retriever/{args.dataset}_{retriever_name}.json'

    if args.retriever == 'bm25':
        retriever = BM25Retriever(index_name=f'{args.dataset}_{len(corpus)}_bm25')
    elif args.retriever == 'facebook/contriever':
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
    elif args.retriever.startswith('BAAI/bge-m3'):
        if args.dataset == 'hotpotqa':
            faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_BAAI_bge-m3_ip_norm.index')
        elif args.dataset == 'musique':
            faiss_index = faiss.read_index('data/musique/musique_BAAI_bge-m3_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_BAAI_bge-m3_ip_norm.index')
        elif args.dataset == 'multihoprag' or args.dataset == 'multihoprag_chunks':
            faiss_index = faiss.read_index(f'data/{args.dataset}/{args.dataset}_BAAI_bge-m3_ip_norm.index')
        retriever = SentenceTransformersRetriever(args.retriever, faiss_index, corpus)
    # large embedding models
    elif args.retriever.startswith('Alibaba-NLP/gte-Qwen2-7B-instruct'):
        if args.dataset == 'hotpotqa':
            faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_Alibaba-NLP_gte-Qwen2-7B-instruct_ip_norm.index')
        elif args.dataset == 'musique':
            faiss_index = faiss.read_index('data/musique/musique_Alibaba-NLP_gte-Qwen2-7B-instruct_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_Alibaba-NLP_gte-Qwen2-7B-instruct_ip_norm.index')
        retriever = LMRetriever(args.retriever, faiss_index, corpus)
    elif args.retriever.startswith('GritLM/GritLM-7B'):
        if args.dataset == 'hotpotqa':
            faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_GritLM_GritLM-7B_ip_norm.index')
        elif args.dataset == 'musique':
            faiss_index = faiss.read_index('data/musique/musique_GritLM_GritLM-7B_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_GritLM_GritLM-7B_ip_norm.index')
        retriever = LMRetriever(args.retriever, faiss_index, corpus)
    elif args.retriever.startswith('nvidia/NV-Embed-v2'):
        if args.dataset == 'hotpotqa':
            faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_nvidia_NV-Embed-v2_ip_norm.index')
        elif args.dataset == 'musique':
            faiss_index = faiss.read_index('data/musique/musique_nvidia_NV-Embed-v2_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_nvidia_NV-Embed-v2_ip_norm.index')
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
        elif args.dataset in ['musique']:
            processed_ids = {sample['id'] for sample in results if 'retrieved' in sample}
        else:
            raise NotImplementedError(f'Dataset {args.dataset} not implemented')
        for sample in results:
            if 'recall' in sample:
                total_recall = {k: total_recall[k] + sample['recall'][str(k)] for k in k_list}
    except Exception as e:
        print('loading results exception', e)
        print(f'Results file {output_path} maybe empty, cannot be loaded.')
        processed_ids = set()

    if len(results) > 0:
        for k in k_list:
            print(f'R@{k}: {total_recall[k] / len(results):.4f} ', end='')
        print()
    if read_existing_data:
        print(f'All samples have been already in the result file ({output_path}), exit.')
        exit(0)

    with ThreadPoolExecutor(max_workers=args.thread) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(process_sample, idx, sample, args, corpus, retriever, client, processed_ids) for idx, sample in enumerate(data)]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc='Parallel RETRIEVING'):
            # idx, recall, retrieved_passages, thoughts, it = future.result()
            try:
                idx, recall, retrieved_passages, thoughts, it = future.result()
            except Exception as e:
                print(f"[ERROR] Exception in thread: {e}")
            
            print("retireved done")
            # print metrics
            for k in k_list:
                total_recall[k] += recall[k]
                print(f'R@{k}: {total_recall[k] / (idx + 1):.4f} ', end='')
            print()
            if args.max_steps > 1:
                print('[ITERATION]', it, '[PASSAGE]', len(retrieved_passages), '[THOUGHT]', thoughts)

            # record results
            results[idx]['retrieved'] = retrieved_passages
            results[idx]['recall'] = recall
            results[idx]['thoughts'] = thoughts

            if idx % 50 == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f)

    # save results
    with open(output_path, 'w') as f:
        json.dump(results, f)
    print(f'Saved results to {output_path}')
    for k in k_list:
        print(f'R@{k}: {total_recall[k] / len(data):.4f} ', end='')
