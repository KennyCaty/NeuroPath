import sys
"""

"""
sys.path.append('.')
import os
import time
import ipdb
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_util import init_langchain_model  # kept for side effects and downstream re-imports
from transformers.hf_argparser import string_to_bool
import argparse
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from threading import Lock
from neuropath import NeuroPath

ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'


def parse_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content by the metadata pattern
    parts = content.split('# METADATA: ')
    parsed_data = []

    for part in parts[1:]:  # Skip the first split as it will be empty
        metadata_section, rest_of_data = part.split('\n', 1)
        metadata = json.loads(metadata_section)
        document_sections = rest_of_data.strip().split('\n\nQ: ')
        document_text = document_sections[0].strip()
        qa_pair = document_sections[1].split('\nA: ')
        question = qa_pair[0].strip()
        answer = qa_pair[1].strip()

        parsed_data.append({
            'metadata': metadata,
            'document': document_text,
            'question': question,
            'answer': answer
        })

    return parsed_data


def retrieve_step(query: str, corpus, top_k: int, rag: NeuroPath, dataset: str, one_shot:bool):
    ranks, scores, candidate_doc_ids, candidate_paths, total_tokens, llm_time_for_one_q = rag.rank_docs(query, top_k=top_k, one_shot=one_shot)
    # print("ranks：", len(ranks))
    # assert False
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        retrieved_passages = []
        for rank in ranks:
            key = list(corpus.keys())[rank]
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    else:
        retrieved_passages = [corpus[rank]['title'] + '\n' + corpus[rank]['text'] for rank in ranks]
    return retrieved_passages, scores, candidate_doc_ids, candidate_paths, total_tokens, llm_time_for_one_q


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
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
    :return: next thought
    """
    prompt_demo = ''
    for sample in few_shot:
        prompt_demo += f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["answer"]}\n\n'

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    messages = ChatPromptTemplate.from_messages([SystemMessage(ircot_reason_instruction + '\n\n' + prompt_demo),
                                                 HumanMessage(prompt_user)]).format_prompt()

    try:
        chat_completion = client.invoke(messages.to_messages())
        response_content = chat_completion.content
    except Exception as e:
        print(e)
        return ''
    return response_content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--index_llm', type=str, default='openai',
                        help="provider of the LLM used for indexing. Only used for locating index files (paths are keyed on index_llm_model).")
    parser.add_argument('--index_llm_model', type=str, default='gpt-4o-mini',
                        help='name of the LLM used for indexing; must match the model that produced the graph files.')
    parser.add_argument('--rag_llm', type=str, default='openai',
                        help="provider of the LLM used at retrieval time (path tracking / query NER).")
    parser.add_argument('--rag_llm_model', type=str, default='gpt-4o-mini',
                        help='name of the LLM used at retrieval time.')
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    parser.add_argument('--max_hop', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=10, help='retrieving k documents at each step')
    parser.add_argument('--dpr_only', type=str, default='f')
    parser.add_argument('--graph_alg', type=str, default='kg_path')
    parser.add_argument('--wo_node_spec', action='store_true')
    parser.add_argument('--force_retry', action='store_true')
    parser.add_argument('--one_shot', type=str, default='f')
    args = parser.parse_args()

    dpr_only = string_to_bool(args.dpr_only)
    one_shot = string_to_bool(args.one_shot)

    rag_llm_model_processed = args.rag_llm_model.replace('/', '_').replace('.', '_')

    rag = NeuroPath(args.dataset,
                    index_llm=args.index_llm, index_llm_model=args.index_llm_model,
                    rag_llm=args.rag_llm, rag_llm_model=args.rag_llm_model,
                    graph_creating_retriever_name=args.retriever,
                    node_specificity=not (args.wo_node_spec), graph_type='facts',
                    dpr_only=dpr_only, graph_alg=args.graph_alg, max_hop=args.max_hop)

    data = json.load(open(f'data/{args.dataset}.json', 'r'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))

    if dpr_only:
        dpr_only_str = 'dpr_only'
    else:
        dpr_only_str = 'neuropath'

    if args.graph_alg == 'kg_path':
        output_path = f'output/retrieved/retrieved_results_{args.dataset}_{dpr_only_str}_{rag.graph_creating_retriever_name_processed}_{rag_llm_model_processed}_top_{args.top_k}'
    else:
        output_path = f'output/retrieved/retrieved_results_{args.dataset}_{dpr_only_str}_{rag.graph_creating_retriever_name_processed}_{rag_llm_model_processed}_top_{args.top_k}_{args.graph_alg}'

    if args.wo_node_spec:
        output_path += 'wo_node_spec'

    output_path += '.json'

    k_list = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 80, 100]
    total_recall = {k: 0 for k in k_list}

    force_retry = args.force_retry

    if force_retry:
        results = []
        processed_ids = set()
    else:
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
                processed_ids = {sample['_id'] for sample in results}
            else:
                processed_ids = {sample['id'] for sample in results}

            for sample in results:
                total_recall = {k: total_recall[k] + sample['recall'][str(k)] for k in k_list}
        except Exception as e:
            print(e)
            print('Results file maybe empty, cannot be loaded.')
            results = []
            processed_ids = set()
            total_recall = {k: 0 for k in k_list}

    print(f'Loaded {len(results)} results from {output_path}')
    if len(results) > 0:
        for k in k_list:
            print(f'R@{k}: {total_recall[k] / len(results):.4f} ', end='')
        print()
        
    # JSON serializer that handles numpy types
    def default_serializer(obj):
        if isinstance(obj, (np.int_, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Type {type(obj)} not serializable")

    

    # ========= multi-thread retrieval ==========
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_sample(sample, processed_ids, args, corpus, rag, k_list):
        """Per-sample processing logic"""
        
        try:
            start_time = time.time()
            
            query = sample['question']
            retrieved_passages, scores, candidate_doc_ids, candidate_paths, total_tokens, llm_time_for_one_q = retrieve_step(query, corpus, args.top_k, rag, args.dataset, one_shot)
            retrieved_passages_dict = {passage: score for passage, score in zip(retrieved_passages, scores)}
            elapsed_time = time.time() - start_time
            sample['processing_time'] = elapsed_time
            # compute Recall
            if args.dataset in ['hotpotqa', 'hotpotqa_train']:
                gold_passages = [item for item in sample['supporting_facts']]
                gold_items = set([item[0] for item in gold_passages])
                retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
            elif args.dataset in ['2wikimultihopqa']:
                gold_passages = [item for item in sample['supporting_facts']]
                gold_items = set([item[0] for item in gold_passages])
                retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
            elif args.dataset in ['musique']:
                gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
                gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
                retrieved_items = retrieved_passages
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
            
            recall = {}
            for k in k_list:
                recall[k] = round(sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items), 4)
            
            # assemble result
            phrases_in_gold_docs = []
            for gold_item in gold_items:
                phrases_in_gold_docs.append(rag.get_phrases_in_doc_str(gold_item))
            
            if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
                sample['supporting_docs'] = [item for item in sample['supporting_facts']]
            elif args.dataset in ['musique']:
                sample['supporting_docs'] = [item for item in sample['paragraphs'] if item['is_supporting']]
                del sample['paragraphs']
            elif args.dataset in ['nq_rear']:
                sample['supporting_docs'] = [item for item in sample['contexts'] if item['is_supporting']]
                del sample['contexts']
            elif args.dataset in ['popqa']:
                sample['supporting_docs'] = [item for item in sample['paragraphs'] if item['is_supporting']]
                del sample['paragraphs']
            elif args.dataset in ['multihoprag', 'multihoprag_chunks']:
                sample['supporting_docs'] = [item for item in sample['evidence_list']]
                del sample['evidence_list']
            
            sample['retrieved'] = retrieved_passages[:10]
            sample['retrieved_scores'] = scores[:10]
            sample['nodes_in_gold_doc'] = phrases_in_gold_docs
            sample['recall'] = recall
            sample['candidate_doc_ids'] = candidate_doc_ids
            sample['candidate_paths'] = candidate_paths      
            
            return sample, total_tokens, llm_time_for_one_q
        
        except Exception as e:
            print(f"Error processing sample {sample.get('_id', sample.get('id'))}: {e}")
            return None

    lock = Lock()  # ensure atomic state updates across threads
    total_processing_time = 0
    valid_sample_count = 0
    total_tokens = 0
    all_llm_time = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for sample_idx, sample in enumerate(data):
            futures.append(executor.submit(process_sample, sample, processed_ids, args, corpus, rag, k_list))
        
        results = []
        total_recall = {k: 0 for k in k_list}
        for future in tqdm(as_completed(futures), total=len(data), desc='retrieval'):
            try:
                result, tokens, llm_time_for_one_q = future.result()
                
                if result is not None:
                    results.append(result)
                    with lock:
                        total_tokens += tokens 
                        total_processing_time += result['processing_time']
                        valid_sample_count += 1
                        all_llm_time += llm_time_for_one_q
                        
                    for k in k_list:
                        total_recall[k] += result['recall'][k]
                    
                    print(f"Processed {len(results)} samples: ", end='')
                    for k in k_list:
                        avg_recall = total_recall[k] / len(results)
                        print(f"R@{k}: {avg_recall:.4f} ", end='')
                    print()
                
                # save every 10 samples
                if len(results) % 10 == 0:
                    with open(output_path, 'w') as f:
                        json.dump(results, f, default=default_serializer)
            
            except Exception as e:
                print(f"Error in a thread: {e}")
        average_processing_time = total_processing_time / valid_sample_count if valid_sample_count > 0 else 0
        
        print(f"Total processing time: {total_processing_time:.4f} seconds")
        print(f"Average processing time per sample: {average_processing_time:.4f} seconds")
        print(f"LLM processing time: {all_llm_time:.4f} seconds")
        print(f"Total tokens: {total_tokens} tokens")
    # ======= multi-thread END ============
  
    # save results
    with open(output_path, 'w') as f:
        json.dump(results, f, default=default_serializer)
    print(f'Saved {len(results)} results to {output_path}')
