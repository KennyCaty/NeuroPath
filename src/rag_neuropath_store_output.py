import sys
"""
"""
sys.path.append('.')
import os
import time
import ipdb
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.langchain_util import init_langchain_model
from transformers.hf_argparser import string_to_bool
import argparse
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from threading import Lock
from neuropath_store_output import NeuroPath

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


def retrieve_step(query: str, corpus, top_k: int, rag: NeuroPath, dataset: str):
    ranks, scores, candidate_doc_ids, candidate_paths, total_tokens, llm_time_for_one_q, train_list = rag.rank_docs(query, top_k=top_k)
    # print("ranks：", len(ranks))
    # assert False
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        retrieved_passages = []
        for rank in ranks:
            key = list(corpus.keys())[rank]
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    else:
        retrieved_passages = [corpus[rank]['title'] + '\n' + corpus[rank]['text'] for rank in ranks]
    return retrieved_passages, scores, candidate_doc_ids, candidate_paths, total_tokens, llm_time_for_one_q, train_list


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
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    parser.add_argument('--prompt', type=str)
    # parser.add_argument('--max_steps', type=int)
    parser.add_argument('--top_k', type=int, default=10, help='retrieving k documents at each step')
    # parser.add_argument('--doc_ensemble', type=str, default='f')
    parser.add_argument('--dpr_only', type=str, default='f')
    parser.add_argument('--graph_alg', type=str, default='kg_path')
    parser.add_argument('--wo_node_spec', action='store_true')
    parser.add_argument('--sim_threshold', type=float, default=0.8)
    # parser.add_argument('--damping', type=float, default=0.1)
    parser.add_argument('--force_retry', action='store_true')
    parser.add_argument('--track_alg', type=str, default='single_step')
    args = parser.parse_args()

    # Please set environment variable OPENAI_API_KEY
    # doc_ensemble = string_to_bool(args.doc_ensemble)
    dpr_only = string_to_bool(args.dpr_only)

    client = init_langchain_model(args.llm, args.llm_model)
    llm_model_name_processed = args.llm_model.replace('/', '_').replace('.', '_')
    # if args.llm_model == 'gpt-3.5-turbo-1106':  # Default OpenIE system
    #     colbert_configs = {'root': f'data/lm_vectors/colbert/{args.dataset}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}
    # else:
    #     colbert_configs = {'root': f'data/lm_vectors/colbert/{args.dataset}_{llm_model_name_processed}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}
    colbert_configs = {'root': f'data/lm_vectors/colbert/{args.dataset}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}

    # graph_type='facts'
    # facts_and_sim
    rag = NeuroPath(args.dataset, args.llm, args.llm_model, args.retriever, node_specificity=not (args.wo_node_spec), graph_type='facts', sim_threshold=args.sim_threshold,
                   colbert_config=colbert_configs, dpr_only=dpr_only, graph_alg=args.graph_alg)

    data = json.load(open(f'data/{args.dataset}.json', 'r'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
    # max_steps = args.max_steps


    if dpr_only:
        dpr_only_str = 'dpr_only'
    else:
        dpr_only_str = 'neuropath'

    if args.graph_alg == 'kg_path':
        output_path = f'output/retrieved/retrieved_results_{args.dataset}_{dpr_only_str}_{rag.graph_creating_retriever_name_processed}_{llm_model_name_processed}_top_{args.top_k}'
    else:
        output_path = f'output/retrieved/retrieved_results_{args.dataset}_{dpr_only_str}_{rag.graph_creating_retriever_name_processed}_{llm_model_name_processed}_top_{args.top_k}_{args.graph_alg}'

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
        
    # 新增
    # 自定义序列化函数 防止int64 和 float32不能被json序列化
    # def default_serializer(obj):
    #     if isinstance(obj, np.int64):
    #         return int(obj)  # 转换为 Python 原生 int 类型
    #     elif isinstance(obj, (np.float32, np.float64)):
    #         return float(obj)  # 将 numpy.float32 和 numpy.float64 转换为 Python float 类型
    #     raise TypeError(f"Type {type(obj)} not serializable")
    def default_serializer(obj):
        # 处理 NumPy 整数类型
        if isinstance(obj, (np.int_, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)  # 转换为 Python 原生 int 类型

        # 处理 NumPy 浮点类型
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)  # 转换为 Python 原生 float 类型

        # 处理 NumPy 布尔类型
        elif isinstance(obj, (np.bool_)):
            return bool(obj)  # 转换为 Python 原生 bool 类型

        # 处理 NumPy 数组
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 将数组转换为列表

        # 如果是其他类型，抛出异常
        raise TypeError(f"Type {type(obj)} not serializable")

    # # ========= 单线程 ==========    
    # count = 0
    # data = data[0:5]
    # for sample_idx, sample in tqdm(enumerate(data), total=len(data), desc='retrieval'):  # for each sample
    #     count += 1
    #     if count ==4:
    #         assert False
    #     # if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
    #     #     sample_id = sample['_id']
    #     # else:
    #     #     sample_id = sample['id']

    #     # if sample_id in processed_ids:
    #     #     continue

    #     query = sample['question']
    #     # all_logs = {}

    #     retrieved_passages, scores, candidate_doc_ids, candidate_paths, tokens, llm_time_for_one_q = retrieve_step(query, corpus, args.top_k, rag, args.dataset)

    #     # it = 1
    #     # all_logs[it] = logs

    #     # thoughts = []
    #     retrieved_passages_dict = {passage: score for passage, score in zip(retrieved_passages, scores)}


    #     # calculate recall
    #     if args.dataset in ['hotpotqa', 'hotpotqa_train']:
    #         gold_passages = [item for item in sample['supporting_facts']]
    #         gold_items = set([item[0] for item in gold_passages])
    #         retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    #     elif args.dataset in ['2wikimultihopqa']:
    #         gold_passages = [item for item in sample['supporting_facts']]
    #         gold_items = set([item[0] for item in gold_passages])
    #         retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    #     else:
    #         gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
    #         gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
    #         retrieved_items = retrieved_passages

    #     # calculate metrics
    #     recall = dict()
    #     print(f'idx: {sample_idx + 1} ', end='')
    #     for k in k_list:
    #         recall[k] = round(sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items), 4)
    #         total_recall[k] += recall[k]
    #         print(f'R@{k}: {total_recall[k] / (sample_idx + 1):.4f} ', end='')
    #     print()


    #     # record results
    #     phrases_in_gold_docs = []
    #     for gold_item in gold_items:
    #         phrases_in_gold_docs.append(rag.get_phrases_in_doc_str(gold_item))

    #     if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
    #         sample['supporting_docs'] = [item for item in sample['supporting_facts']]
    #     else:
    #         sample['supporting_docs'] = [item for item in sample['paragraphs'] if item['is_supporting']]
    #         del sample['paragraphs']

    #     sample['retrieved'] = retrieved_passages[:10]
    #     sample['retrieved_scores'] = scores[:10]
    #     sample['nodes_in_gold_doc'] = phrases_in_gold_docs
    #     sample['recall'] = recall
    #     sample['candidate_doc_ids'] = candidate_doc_ids
    #     sample['candidate_paths'] = candidate_paths
    #     # first_log = all_logs[1]
    #     # for key in first_log.keys():
    #     #     sample[key] = first_log[key]
    #     # sample['thoughts'] = thoughts
    #     results.append(sample) 
    #     if (sample_idx + 1) % 10 == 0:
    #         with open(output_path, 'w') as f:
    #             json.dump(results, f,  default=default_serializer)

    # ========= 多线程 ==========
    # data = data[:1]
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # 读入之前提取KG信息的文件
    with open(glob(f"output/openie_{args.dataset}_results_ner_{args.llm_model}_*.json")[0], 'r') as f:
        openie = json.load(f)['docs']
    
    def process_sample(sample, processed_ids, args, corpus, rag, k_list):
        """单个样本的处理逻辑"""
        
        try:
            start_time = time.time()  # 记录开始时间
                
            # if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
            #     sample_id = sample['_id']
            # else:
            #     sample_id = sample['id']
            
            # if sample_id in processed_ids:
            #     return None  # 跳过已处理的样本
            
            query = sample['question']
            # all_logs = {}
            retrieved_passages, scores, candidate_doc_ids, candidate_paths, total_tokens, llm_time_for_one_q, train_list = retrieve_step(query, corpus, args.top_k, rag, args.dataset)
            # it = 1
            # all_logs[it] = logs
            # thoughts = []
            retrieved_passages_dict = {passage: score for passage, score in zip(retrieved_passages, scores)}
            elapsed_time = time.time() - start_time  # 记录结束时间并计算耗时
            sample['processing_time'] = elapsed_time  # 将处理时间添加到样本中
            # 计算 Recall
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
            
            recall = {}
            for k in k_list:
                recall[k] = round(sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items), 4)
            
            # 构造结果
            phrases_in_gold_docs = []
            for gold_item in gold_items:
                phrases_in_gold_docs.append(rag.get_phrases_in_doc_str(gold_item))
            
            if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
                sample['supporting_docs'] = [item for item in sample['supporting_facts']]
            elif args.dataset in ['musique']:
                sample['supporting_docs'] = [item for item in sample['paragraphs'] if item['is_supporting']]
                del sample['paragraphs']
            elif args.dataset in ['musique']:
                sample['supporting_docs'] = [item for item in sample['contexts'] if item['is_supporting']]
                del sample['contexts']
            
            sample['retrieved'] = retrieved_passages[:10]
            sample['retrieved_scores'] = scores[:10]
            sample['nodes_in_gold_doc'] = phrases_in_gold_docs
            sample['recall'] = recall
            sample['candidate_doc_ids'] = candidate_doc_ids
            sample['candidate_paths'] = candidate_paths      
            
            # ner_descriptions = []
            # for doc_id in candidate_doc_ids:
            #     doc = openie[doc_id]
            #     ner_description = doc['ner_description']
            #     ner_descriptions.append(ner_description)
            # sample['ner_descriptions'] = ner_descriptions  
            

            
            return sample, total_tokens, llm_time_for_one_q, train_list
        
        except Exception as e:
            # 捕获异常并记录日志
            print(f"Error processing sample {sample.get('_id', sample.get('id'))}: {e}")
            return None

    lock = Lock() # 保证修改变量的操作是原子的
    total_processing_time = 0  # 总处理时间
    valid_sample_count = 0     # 有效样本数量
    total_tokens = 0           # 总消耗tokens
    all_llm_time = 0
    
    # 如果 output_path 存在 直接计算指标
    # if os.path.exists(output_path):
        
    
    # 主函数
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for sample_idx, sample in enumerate(data):
            futures.append(executor.submit(process_sample, sample, processed_ids, args, corpus, rag, k_list))
        
        results = []
        ft_data = []
        total_recall = {k: 0 for k in k_list}
        for future in tqdm(as_completed(futures), total=len(data), desc='retrieval'):
            try:
                result, tokens, llm_time_for_one_q, train_list = future.result()
                
                if result is not None:
                    with lock:
                        results.append(result)
                        total_tokens += tokens 
                        # 更新总处理时间和有效样本数量
                        total_processing_time += result['processing_time']
                        valid_sample_count += 1
                        all_llm_time += llm_time_for_one_q
                        ft_data.extend(train_list)
                        
                        # 更新 Recall 指标
                        for k in k_list:
                            total_recall[k] += result['recall'][k]
                    
                    # 打印实时 Recall
                    print(f"Processed {len(results)} samples: ", end='')
                    for k in k_list:
                        avg_recall = total_recall[k] / len(results)
                        print(f"R@{k}: {avg_recall:.4f} ", end='')
                    print()
                
                # 每 10 个样本保存一次结果
                if len(results) % 10 == 0:
                    with open(output_path, 'w') as f:
                        json.dump(results, f, default=default_serializer)
            
                    with open('./ft_data.json', 'w') as f:
                        json.dump(ft_data, f)
            except Exception as e:
                # 捕获 Future 中的异常
                print(f"Error in a thread: {e}")
        
        
        # 计算平均处理时间
        average_processing_time = total_processing_time / valid_sample_count if valid_sample_count > 0 else 0
        
        # 打印总时间和平均时间
        print(f"Total processing time: {total_processing_time:.4f} seconds")
        print(f"Average processing time per sample: {average_processing_time:.4f} seconds")
        print(f"LLM processing time: {all_llm_time:.4f} seconds")
        print(f"Total tokens: {total_tokens} tokens")
    # ======= 多线程 END ============
  
    # save results
    with open(output_path, 'w') as f:
        json.dump(results, f, default=default_serializer)
    print(f'Saved {len(results)} results to {output_path}')
