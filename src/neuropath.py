import json
import logging
import os
import _pickle as pickle
from collections import defaultdict
from glob import glob

import igraph as ig
import numpy as np
import pandas as pd
import torch
# from colbert import Searcher
# from colbert.data import Queries
# from colbert.infra import RunConfig, Run, ColBERTConfig
# from src.colbertv2_indexing import colbertv2_index
from tqdm import tqdm

from src.langchain_util import init_langchain_model, LangChainModel
from src.lm_wrapper.util import init_embedding_model
from src.query_ner_vtp_parallel import  ner_vtp_extraction
from src.processing import processing_phrases, min_max_normalize
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import time

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'

COLBERT_CKPT_DIR = "exp/colbertv2.0"



logging.getLogger("httpx").setLevel(logging.WARNING)

import re

def extract_json_dict(text):
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}'
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return {}
    else:
        return {}




class NeuroPath:

    def __init__(self, corpus_name='hotpotqa', extraction_model='openai', extraction_model_name='gpt-4o-mini',
                 graph_creating_retriever_name='facebook/contriever', extraction_type='ner', graph_type='facts', sim_threshold=0.8, node_specificity=True,
                 colbert_config=None, dpr_only=False, graph_alg='kg_path', corpus_path=None,
                 qa_model: LangChainModel = None, linking_retriever_name=None):


        self.corpus_name = corpus_name
        self.extraction_model_name = extraction_model_name
        self.extraction_model_name_processed = extraction_model_name.replace('/', '_')
        self.client = init_langchain_model(extraction_model, extraction_model_name)
        assert graph_creating_retriever_name
        if linking_retriever_name is None:
            linking_retriever_name = graph_creating_retriever_name
        self.graph_creating_retriever_name = graph_creating_retriever_name  # 'colbertv2', 'facebook/contriever', or other HuggingFace models 
        self.graph_creating_retriever_name_processed = graph_creating_retriever_name.replace('/', '_').replace('.', '')
        self.linking_retriever_name = linking_retriever_name
        self.linking_retriever_name_processed = linking_retriever_name.replace('/', '_').replace('.', '')

        self.extraction_type = extraction_type
        self.graph_type = graph_type
        self.phrase_type = 'ents_only_lower_preprocess'
        self.sim_threshold = sim_threshold
        self.node_specificity = node_specificity
        if colbert_config is None:
            self.colbert_config = {'root': f'data/lm_vectors/colbert/{corpus_name}',
                                   'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}
        else:
            self.colbert_config = colbert_config  # a dict, 'root', 'doc_index_name', 'phrase_index_name'

        self.graph_alg = graph_alg

        self.version = 'v3'
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        try:
            self.named_entity_cache = pd.read_csv('output/{}_queries.named_entity_output.tsv'.format(self.corpus_name), sep='\t')
        except Exception as e:
            print("named_entity read error, ",e)
            self.named_entity_cache = pd.DataFrame([], columns=['query', 'triples'])

        if 'query' in self.named_entity_cache:
            self.named_entity_cache = {row['query']: eval(row['triples']) for i, row in
                                       self.named_entity_cache.iterrows()}
        elif 'question' in self.named_entity_cache:
            self.named_entity_cache = {row['question']: eval(row['triples']) for i, row in self.named_entity_cache.iterrows()}


        self.embed_model = init_embedding_model(self.linking_retriever_name)
        self.dpr_only = dpr_only
        self.corpus_path = corpus_path
        ## 新增
        self.fact_json = json.load(open('output/{}_{}_graph_clean_facts_chatgpt_openIE.{}_{}_gpt-4o-mini.{}.subset.json'.format(corpus_name, graph_type, self.phrase_type, extraction_type, self.version), 'r'))

        # Loading Important Corpus Files
        if not self.dpr_only:
            self.load_index_files()

            # Construct Graph
            self.build_graph()

            # Loading Node Embeddings
            self.load_node_vectors()
        else:
            self.load_corpus()

        if (dpr_only or graph_alg=='kg_path') and self.linking_retriever_name not in ['colbertv2', 'bm25']:
            # Loading Doc Embeddings
            self.get_dpr_doc_embedding()

        if self.linking_retriever_name == 'colbertv2':
            if self.dpr_only is False or self.doc_ensemble:
                colbertv2_index(self.phrases.tolist(), self.corpus_name, 'phrase', self.colbert_config['phrase_index_name'], overwrite=True)
                with Run().context(RunConfig(nranks=1, experiment="phrase", root=self.colbert_config['root'])):
                    config = ColBERTConfig(root=self.colbert_config['root'], )
                    self.phrase_searcher = Searcher(index=self.colbert_config['phrase_index_name'], config=config, verbose=0)
            if self.doc_ensemble or dpr_only:
                colbertv2_index(self.dataset_df['paragraph'].tolist(), self.corpus_name, 'corpus', self.colbert_config['doc_index_name'], overwrite=True)
                with Run().context(RunConfig(nranks=1, experiment="corpus", root=self.colbert_config['root'])):
                    config = ColBERTConfig(root=self.colbert_config['root'], )
                    self.corpus_searcher = Searcher(index=self.colbert_config['doc_index_name'], config=config, verbose=0)

        self.statistics = {}
        self.ensembling_debug = []
        if qa_model is None:
            qa_model = LangChainModel('openai', 'gpt-3.5-turbo')
        self.qa_model = init_langchain_model(qa_model.provider, qa_model.model_name)

        ## ===== 新增  phrase faiss index  =====
        import faiss
         # 归一化 embedding（用于余弦相似度）
        faiss.normalize_L2(self.kb_node_phrase_embeddings)
        # 创建GPU资源
        resource = faiss.StandardGpuResources()
        # 创建GPU索引
        index = faiss.IndexFlatIP(self.kb_node_phrase_embeddings.shape[1])
        # 将索引转移到GPU
        gpu_index = faiss.index_cpu_to_gpu(resource, 0, index)
        # 添加 embedding 到索引
        gpu_index.add(self.kb_node_phrase_embeddings)
        # 设置检索参数
        k = 5  # 检索最相似的 5 个（不包括自身）

        # 执行搜索
        distances, indices = gpu_index.search(self.kb_node_phrase_embeddings, k)

        # 过滤相似度低于 0.85 的结果 # Pseudo-Coreference Resolution
        self.coreference_resolution = []
        for i in range(len(self.kb_node_phrase_embeddings)):
            valid_indices = [idx for idx, dist in zip(indices[i], distances[i]) if dist >= 0.8 and idx != i]
            self.coreference_resolution.append(valid_indices)

        # # 输出结果
        # for i, neighbors in enumerate(filtered_results[:20]):
        #     print(f"Embedding {i} 的相似 embedding: {neighbors}")
        ## ======= END ============



   
    def llm_path_track(self, path_prompt, query, vtp=None):
        start_time = time.time()
        # print(vtp)
        # paths = path_dict['paths']
        sys_prompt = """To answer a given query, you need to select a set of valid clue paths to form a chain of reasoning and determine whether some paths need to be expanded with more information to help answer the query.
# Explanation
Valid paths can provide intermediate reasoning steps or evidence to help answer the query. Paths may be redundant, filter them out.
A path that contains the <expandable> tag is an expandable path, which identifies a phrase that can be expanded with additional information. 
If information is sufficient or no more valid information can be added, stop expanding.
Return the required JSON object.
# JSON object format
{
    "current_chain": "A string. You should think step by step. Try to form the current chain of reasoning.",
    "valid_ids": [List of valid path IDs(int) (sort by helpfulness to query)],
    "expansion_requirements": "A string. If need expand, provide the specific requirements of the expansion. Otherwise set to an empty string.",
    "need_expand_ids": [List of path IDs(int) that need to be expanded. (if any)],
    "continue": 0 or 1 (0 = stop expanding, 1 = continue expanding)
}
"""
        prompt = f"""Query: {query}
Paths:\n"""
        prompt += path_prompt + '\nOutput:\n'
        # print(prompt)
        # assert False
        response = ''
        try :
            chat_completion = self.client.invoke(
                [SystemMessage(sys_prompt), HumanMessage(prompt)],
                temperature=0.0, 
                response_format={"type": "json_object"}
                )
            response = chat_completion.content
            total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
            # print(response)
            response = extract_json_dict(response)
            # response = json.loads(response)
            response['valid_ids']
            # response['continue']
            # response['need_expand_ids']

        except Exception as e:
            print(e, "llm response error")
            if response=='':
                print("json.loads error, llms response is not a json")
            else:
                print(response)
            response = {
                'valid_ids': [],
                'continue': 0,
                'need_expand_ids': [],
                'current_chain': '',
                'expansion_requirements': ''
            }
            total_tokens = 0
        # assert False
        llm_time = time.time() - start_time
        return  response, total_tokens, llm_time
    

    def expand_seed(self, seed_phrase_ids):
        expanded_ids = [item for item in seed_phrase_ids]
        # print("seed phrases: ", seed_phrase_ids)
        for idx in seed_phrase_ids:
            expand = self.coreference_resolution[idx]
            # print("idx: ", idx)
            # print("expand: ", expand)
            # assert False
            expanded_ids.extend(expand)
            # print("expanded_ids: ", expanded_ids)
        # assert False
        return list(set(expanded_ids))
    
    
    def pruning(self, old_path_len, filter_q, path_dict, filter_k=30):
        
        # print("filter_q: ", filter_q)
        # start_time = time.time() 
        if len(path_dict['paths']) - old_path_len <= filter_k:
            return path_dict
        
        # print("触发剪枝")
        # 只剪枝扩展的部分
        paths = []
        for i in  range(old_path_len, len(path_dict['paths'])):
            paths.append(path_dict['paths'][i].replace("->", " "))
        
        # paths = path_dict['paths']
        paths = [filter_q] + paths
        path_embeddings = self.embed_model.encode(paths, normalize_embeddings=True, device='cuda', show_progress_bar=False, batch_size=256
            )
        # 
        
        
        
        query_embedding = path_embeddings[0].reshape(-1, 1)  
        path_embeddings = path_embeddings[1:]
        similarity_scores = np.dot(path_embeddings, query_embedding).flatten()
        topk_indices = np.argsort(similarity_scores)[-filter_k:][::-1]  
        
        indices = list(range(old_path_len))
        for i in topk_indices:
            indices.append(i+old_path_len)
        
        
        # print("Top-k 最大值索引:", topk_indices)
        path_dict['paths'] = [path_dict['paths'][i] for i in indices]
        path_dict['link_phrases'] = [path_dict['link_phrases'][i] for i in indices]
        path_dict['visited_docs'] = [path_dict['visited_docs'][i] for i in indices]
        # used_time = time.time() - start_time
        # print("pruning used time: ",used_time)
        return path_dict
    

    def Expand_by_llm(self, path_dict, query_embedding, query, vtp=None): 
        total_tokens = 0
        # 第一轮
        if path_dict['paths'] == []:
            new_link_phrases = []
            expand_phrase_ids = self.expand_seed(path_dict['link_phrases'])
            for link_phrase_id in expand_phrase_ids:
                if self.phrases[link_phrase_id] == 'NaN' or self.phrases[link_phrase_id] == 'Null' or self.phrases[link_phrase_id] == '':  # 为空，不需要扩展
                    continue
                # 找到这个phrase id 对应的doc raw
                doc_row = self.phrase_to_doc_mat[[link_phrase_id]].toarray()[0]
                # 找到非零文档元素的索引列表
                non_zero_doc_ids = np.nonzero(doc_row)[0].tolist()
                # 筛选包含phrase的子路径
                for doc_id in non_zero_doc_ids:
                    target_facts_ids_list = self.docs_to_facts_mat[[doc_id]].toarray()[0]
                    non_zero_target_facts_ids = np.nonzero(target_facts_ids_list)[0].tolist()
                    # 收集路径
                    
                    for facts_id in non_zero_target_facts_ids:
                        if self.facts_to_phrases_mat[facts_id, link_phrase_id] != 0:
                            triple_json = self.fact_json[facts_id]
                            # triple_semantics = '[' + triple_json['head'] + ']' + ' -> ' + '(' + triple_json['relation'] + ')' + ' -> ' + '[' + triple_json['tail'] + ']' + '; '
                            triple_semantics =  triple_json['head'] + '->' + triple_json['relation'] + '->' + triple_json['tail'] + '; '
                            if triple_semantics in path_dict['paths']:
                                # print("路径已存在")
                                continue
                            head_id = self.kb_node_phrase_to_id[triple_json['head']]
                            tail_id = self.kb_node_phrase_to_id[triple_json['tail']]
                            # 拿到子路径的另一个连接的phrase
                            if head_id==link_phrase_id:
                                new_link_phrase_id = tail_id
                                # assert tail_id != link_phrase_id
                            else:
                                new_link_phrase_id = head_id
                                # assert tail_id == link_phrase_id
                            new_link_phrases.append(new_link_phrase_id)
                            path_dict['paths'].append(triple_semantics)
                            path_dict['visited_docs'].append([doc_id])
                            
            path_dict['link_phrases'] = new_link_phrases
            
            # 相似度剪枝
            path_dict = self.pruning(0, vtp, path_dict)
            
            # print(path_dict)
            path_prompt = ''
            for i in range(len(path_dict['paths'])):
                path_prompt += f"{i}: " + path_dict['paths'][i] + ' <expandable>: [' + self.phrases[path_dict['link_phrases'][i]] + ']' + '\n'
            
            response_json, tokens, llm_time = self.llm_path_track(path_prompt, query, vtp)
            total_tokens += tokens       
            
            try:
                new_path_ids_list = response_json['valid_ids']
                path_dict['valid_paths'] = [path_dict['paths'][i] for i in new_path_ids_list]
                path_dict['valid_visited_docs'] = [path_dict['visited_docs'][i] for i in new_path_ids_list]  
                expand_path_list = response_json['need_expand_ids']
                new_path_ids_list.extend(expand_path_list)
                new_path_ids_list = list(set(new_path_ids_list))
                flag = response_json['continue']
                path_dict['need_expand'] = [path_dict['paths'][i] for i in expand_path_list]
                path_dict['paths'] = [path_dict['paths'][i] for i in new_path_ids_list]
                path_dict['link_phrases'] = [path_dict['link_phrases'][i] for i in new_path_ids_list]
                path_dict['visited_docs'] = [path_dict['visited_docs'][i] for i in new_path_ids_list]  
                
                # 
                path_dict['current_chain'] = response_json['current_chain']
                path_dict['expansion_requirements'] = response_json['expansion_requirements']

                
                # print("第二轮 llms返回path_dict: \n", path_dict)
            except Exception as e:
                print(e, "response parsing error")
                print(path_dict)
                print(response_json)
                # assert False
                path_dict['valid_paths'] = []
                path_dict['visited_docs'] = []
                path_dict['valid_visited_docs'] = []
                path_dict['need_expand'] = []
                path_dict['paths'] = []
                path_dict['current_chain'] = ''
                path_dict['expansion_requirements'] = ''

                flag = 0
            return path_dict, flag, total_tokens, llm_time
           
           
           
        # 之后的轮数
        else:
            # 上一轮的有效路径数量
            old_path_len = len(path_dict['paths'])
            
            new_link_phrases = []
            new_paths = []
            new_visited_docs = []
            locked_path = []
            for i in range(len(path_dict['paths'])):
                path = path_dict['paths'][i]
                if path not in path_dict['need_expand']:
                    # print(f"路径:{path}被锁住")
                    locked_path.append(path)
            for i in range(len(path_dict['paths'])):  # 对于每一个可扩展路径而言，分别扩充路径
                # 如果路径可扩展
                if path_dict['paths'][i] not in locked_path:
                    path = path_dict['paths'][i]
                    seed_phrase_id = path_dict['link_phrases'][i]
                    # 扩展相似短语 （共指消解）
                    expand_phrase_ids = self.expand_seed([seed_phrase_id])
                    visited_docs = path_dict['visited_docs'][i]
                    
                    for link_phrase_id in expand_phrase_ids:
                        if self.phrases[link_phrase_id] == 'NaN' or self.phrases[link_phrase_id] == 'Null' or self.phrases[link_phrase_id] == '':  # 为空，不需要扩展
                            continue
                        # 找到这个phrase id 对应的doc raw
                        doc_row = self.phrase_to_doc_mat[[link_phrase_id]].toarray()[0]
                        # 找到非零文档元素的索引列表
                        non_zero_doc_ids = np.nonzero(doc_row)[0].tolist()
                        # 找到link_phrase对应的文档（未访问过）
                        for doc_id in non_zero_doc_ids:
                            # if doc_id not in visited_docs:
                            target_facts_ids_list = self.docs_to_facts_mat[[doc_id]].toarray()[0]
                            non_zero_target_facts_ids = np.nonzero(target_facts_ids_list)[0].tolist()
                            # 收集路径
                            for facts_id in non_zero_target_facts_ids:
                                if self.facts_to_phrases_mat[facts_id, link_phrase_id] != 0:
                                    triple_json = self.fact_json[facts_id]
                                    # triple_semantics = '[' + triple_json['head'] + ']' + ' -> ' + '(' + triple_json['relation'] + ')' + ' -> ' + '[' + triple_json['tail'] + ']' + '; '
                                    triple_semantics =  triple_json['head'] + '->' + triple_json['relation'] + '->' + triple_json['tail'] + '; '
                                    head_id = self.kb_node_phrase_to_id[triple_json['head']]
                                    tail_id = self.kb_node_phrase_to_id[triple_json['tail']]
                                    if triple_semantics in path_dict['paths'] or triple_semantics in path_dict['need_expand']:
                                        # print("路径已存在")
                                        continue
                                    # 拿到子路径的另一个连接的phrase
                                    if head_id==link_phrase_id:
                                        new_link_phrase_id = tail_id
                                        # assert tail_id != link_phrase_id
                                    else:
                                        new_link_phrase_id = head_id
                                        # assert tail_id == link_phrase_id
                                    new_link_phrases.append(new_link_phrase_id)
                                    path_i = path
                                    path_i += triple_semantics
                                    new_paths.append(path_i)
                                    new_visited_docs.append(visited_docs + [doc_id])
            path_dict['link_phrases'].extend(new_link_phrases)
            path_dict['paths'].extend(new_paths)
            path_dict['visited_docs'].extend(new_visited_docs)
            
            # 相似度剪枝
            path_dict = self.pruning(old_path_len, vtp, path_dict)
            
            path_prompt = ''
            for i in range(len(path_dict['paths'])):
                if path_dict['paths'][i] in locked_path: # 把无需扩展的锁住
                    # path_prompt += f"{i}: " + path_dict['paths'][i] + ' <locked>' + '\n'
                    path_prompt += f"{i}: " + path_dict['paths'][i] + '\n'
                elif path_dict['paths'][i] in path_dict['need_expand']: # 把已经扩展的锁住
                    # path_prompt += f"{i}: " + path_dict['paths'][i] + ' <locked>' + '\n'
                    path_prompt += f"{i}: " + path_dict['paths'][i] + '\n'
                else:
                    path_prompt += f"{i}: " + path_dict['paths'][i] + ' <expandable>: [' + self.phrases[path_dict['link_phrases'][i]] + ']' + '\n'
                    
            
            # print("第二轮 初始path_dict: \n", path_dict)
            response_json, tokens, llm_time = self.llm_path_track(path_prompt, query, vtp)
            total_tokens += tokens
            try:
                new_path_ids_list = response_json['valid_ids']
                path_dict['valid_paths'] = [path_dict['paths'][i] for i in new_path_ids_list]
                path_dict['valid_visited_docs'] = [path_dict['visited_docs'][i] for i in new_path_ids_list]  
                flag = response_json['continue']
                expand_path_list = response_json['need_expand_ids']
                new_path_ids_list.extend(expand_path_list)
                new_path_ids_list = list(set(new_path_ids_list))
                path_dict['need_expand'] = [path_dict['paths'][i] for i in expand_path_list]
                # print("需要扩展的路径: ", path_dict['need_expand'] )
                path_dict['paths'] = [path_dict['paths'][i] for i in new_path_ids_list]
                # print("当先所有有效路径: ", path_dict['paths'] )
                path_dict['link_phrases'] = [path_dict['link_phrases'][i] for i in new_path_ids_list]
                # print("有效路径连接的节点: ", path_dict['link_phrases'] )
                path_dict['visited_docs'] = [path_dict['visited_docs'][i] for i in new_path_ids_list]         
                # print("第二轮 llms返回path_dict: \n", path_dict)
                
                # 
                path_dict['current_chain'] = response_json['current_chain']
                path_dict['expansion_requirements'] = response_json['expansion_requirements']

            except Exception as e:
                print(e, "response error")
                path_dict['valid_paths'] = []
                path_dict['paths'] =[]
                path_dict['visited_docs'] = []
                path_dict['valid_visited_docs'] = []
                path_dict['need_expand'] = []
                path_dict['current_chain'] = ''
                path_dict['expansion_requirements'] = ''

                flag = 0
            return path_dict, flag, total_tokens, llm_time

        
        
    def get_path_filtered_docs(self, query_embedding, seed_phrase_ids, query, vtp=None):
        unique_phrase_ids_set = set(seed_phrase_ids)
        # 初始化
        path_dict = {}
        path_dict['paths'] = []
        path_dict['link_phrases'] = list(unique_phrase_ids_set)
        path_dict['visited_docs'] = []  # 即访问的doc路径
        path_dict['need_expand_ids'] = []
        flag = 1
        max_layer = 2
        count = 0
        total_tokens = 0
        cur_vtp = query
        
        all_thought = ''
        llm_time_for_one_q = 0
        while count < max_layer and flag==1:
            count += 1
            path_dict, flag, tokens, llm_time = self.Expand_by_llm(path_dict, query_embedding, query, cur_vtp)
            llm_time_for_one_q += llm_time
            total_tokens += tokens
            all_thought = path_dict['current_chain'] + path_dict['expansion_requirements']
            cur_vtp =  path_dict['expansion_requirements']

            if flag==0 and path_dict['valid_paths'] != []:
                print("The LLM determines the path.")
            elif path_dict['valid_paths'] == [] and path_dict['need_expand'] == []:
                print("The LLM failed to find a path.")
                flag = 0
            
        
        #取文档id
        candidate_docs = []
        candidate_paths = path_dict['valid_paths']
        visited_docs = path_dict['valid_visited_docs']
        for each_path_docs in visited_docs:
            for doc in each_path_docs:
                if doc not in candidate_docs:
                    candidate_docs.append(doc)

        # print("候选文档：", candidate_docs)
        # assert False
        return candidate_docs, candidate_paths, total_tokens, all_thought, llm_time_for_one_q
       
 
    
    def rank_docs(self, query: str, top_k, seed=1, expand_num=1):  # 
        """
        Rank documents based on the query
        @param query: the input phrase
        @param top_k: the number of documents to return
        @return: the ranked document ids and their scores
        """

        assert isinstance(query, str), 'Query must be a string'
        # query_ner_list, virtual_target_path = self.query_ner(query)
        query_ner_list = self.query_ner(query)


        if self.graph_alg=='kg_path':

            if len(query_ner_list) > 0: # 使用Path
                query_ner_embeddings = self.embed_model.encode(query_ner_list, normalize_embeddings=True, device='cuda', show_progress_bar=False)
                # Get Closest Entity Nodes
                prob_vectors = np.dot(query_ner_embeddings, self.kb_node_phrase_embeddings.T)  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)
                linked_phrase_ids = []  # Seed Phrases

                for prob_vector in prob_vectors:
                    phrase_id = np.argmax(prob_vector)  # the phrase with the highest similarity  只取了相似性最大的一个id
                    linked_phrase_ids.append(phrase_id)
                
                
                # 拿到新排名的doc id
                candidate_doc_ids, candidate_paths, total_tokens, all_thought, llm_time_for_one_q = self.get_path_filtered_docs(None, linked_phrase_ids, query, None)
                
                query_embedding = self.embed_model.encode([query+all_thought], normalize_embeddings=True, device='cuda', show_progress_bar=False)
                # assert False
                
                query_doc_scores = np.dot(self.doc_embedding_mat, query_embedding.T)
                query_doc_scores = query_doc_scores.T[0]

                doc_ids = []
                doc_scores = []
                # 拿到了所有文档的根据向量相似度的分数
                doc_prob = query_doc_scores
                # 排序后的id和分数
                sorted_doc_ids = np.argsort(doc_prob)[::-1]
                sorted_scores = doc_prob[sorted_doc_ids]
                
                for t in candidate_doc_ids:
                    doc_ids.append(t)
                    doc_scores.append(1) # 占位
                # for idx in range(len(doc_prob)):  # 补足后面的检索文档，方便计算top k的召回率
                for idx in range(100):  # 补足后面的检索文档，方便计算top k的召回率
                    if sorted_doc_ids[idx] not in doc_ids:
                        doc_ids.append(sorted_doc_ids[idx])
                        doc_scores.append(sorted_scores[idx])
            else:  # 直接使用相似性检索
                query_embedding = self.embed_model.encode([query], normalize_embeddings=True, device='cuda', show_progress_bar=False)
                # query_embedding = self.embed_model.encode([virtual_target_path.replace("->", " ")], normalize_embeddings=True, device='cuda', show_progress_bar=False)
                # assert False
                query_doc_scores = np.dot(self.doc_embedding_mat, query_embedding.T)
                query_doc_scores = query_doc_scores.T[0]


                doc_ids = []
                doc_scores = []
                # 拿到了所有文档的根据向量相似度的分数
                doc_prob = query_doc_scores
                # 排序后的id和分数
                # sorted_doc_ids = np.argsort(doc_prob, kind='mergesort')[::-1]
                sorted_doc_ids = np.argsort(doc_prob)[::-1]
                sorted_scores = doc_prob[sorted_doc_ids]
                
                #
                doc_ids = sorted_doc_ids
                doc_scores = sorted_scores
                
                candidate_doc_ids = []
                candidate_paths = []
                total_tokens = 0
                llm_time_for_one_q = 0
        return doc_ids[:top_k], doc_scores[:top_k], candidate_doc_ids, candidate_paths, total_tokens, llm_time_for_one_q
    
    

    def query_ner(self, query):
        if self.dpr_only:
            query_ner_list = []
        else:
            # Extract Entities And Virtual Target Path
            try:
                if query in self.named_entity_cache:
                    query_ner_list = self.named_entity_cache[query]['named_entities']
                    
                else:
                    print("None query named_entity_cache")
                    query_ner_json, total_tokens = ner_vtp_extraction(self.client, query)
                    query_ner_list = eval(query_ner_json)['named_entities']

                query_ner_list = [processing_phrases(p) for p in query_ner_list]
            except:
                self.logger.error('Error in Query NER')
                query_ner_list = []
        return query_ner_list

    def get_neighbors(self, prob_vector, max_depth=1):

        initial_nodes = prob_vector.nonzero()[0]
        min_prob = np.min(prob_vector[initial_nodes])

        for initial_node in initial_nodes:
            all_neighborhood = []

            current_nodes = [initial_node]

            for depth in range(max_depth):
                next_nodes = []

                for node in current_nodes:
                    next_nodes.extend(self.g.neighbors(node))
                    all_neighborhood.extend(self.g.neighbors(node))

                current_nodes = list(set(next_nodes))

            for i in set(all_neighborhood):
                prob_vector[i] += 0.5 * min_prob

        return prob_vector

    def load_corpus(self):
        if self.corpus_path is None:
            self.corpus_path = 'data/{}_corpus.json'.format(self.corpus_name)
        assert os.path.isfile(self.corpus_path), 'Corpus file not found'
        self.corpus = json.load(open(self.corpus_path, 'r'))
        self.dataset_df = pd.DataFrame()
        self.dataset_df['paragraph'] = [p['title'] + '\n' + p['text'] for p in self.corpus]

    def load_index_files(self):
        index_file_pattern = 'output/openie_{}_results_{}_{}_*.json'.format(self.corpus_name, self.extraction_type, self.extraction_model_name_processed)
        possible_files = glob(index_file_pattern)
        if len(possible_files) == 0:
            self.logger.critical(f'No extraction files found: {index_file_pattern} ; please check if working directory is correct or if the extraction has been done.')
            return
        max_samples = np.max(
            [int(file.split('{}_'.format(self.extraction_model_name_processed))[1].split('.json')[0]) for file in possible_files])
        extracted_file = json.load(open(
            'output/openie_{}_results_{}_{}_{}.json'.format(self.corpus_name, self.extraction_type, self.extraction_model_name_processed, max_samples),
            'r'))

        self.extracted_triples = extracted_file['docs']

        if self.corpus_name == 'hotpotqa':
            self.dataset_df = pd.DataFrame([p['passage'].split('\n')[0] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        if self.corpus_name == 'hotpotqa_train':
            self.dataset_df = pd.DataFrame([p['passage'].split('\n')[0] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        elif 'musique' in self.corpus_name:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        elif self.corpus_name == '2wikimultihopqa':
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
            self.dataset_df['title'] = [s['title'] for s in self.extracted_triples]
        elif 'case_study' in self.corpus_name:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]
        else:
            self.dataset_df = pd.DataFrame([p['passage'] for p in self.extracted_triples])
            self.dataset_df['paragraph'] = [s['passage'] for s in self.extracted_triples]

        if self.extraction_model_name != 'gpt-3.5-turbo-1106':
            self.extraction_type = self.extraction_type + '_' + self.extraction_model_name_processed
        self.kb_node_phrase_to_id = pickle.load(open(
            'output/{}_{}_graph_phrase_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type,
                                                                      self.extraction_type, self.version), 'rb'))
        self.lose_fact_dict = pickle.load(open(
            'output/{}_{}_graph_fact_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type,
                                                                    self.extraction_type, self.version), 'rb'))

        try:
            self.relations_dict = pickle.load(open(
                'output/{}_{}_graph_relation_dict_{}_{}_{}.{}.subset.p'.format(
                    self.corpus_name, self.graph_type, self.phrase_type,
                    self.extraction_type, self.graph_creating_retriever_name_processed, self.version), 'rb'))
        except:
            pass

        self.lose_facts = list(self.lose_fact_dict.keys())
        self.lose_facts = [self.lose_facts[i] for i in np.argsort(list(self.lose_fact_dict.values()))]
        self.phrases = np.array(list(self.kb_node_phrase_to_id.keys()))[np.argsort(list(self.kb_node_phrase_to_id.values()))]

        self.docs_to_facts = pickle.load(open(
            'output/{}_{}_graph_doc_to_facts_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type,
                                                                       self.extraction_type, self.version), 'rb'))
        self.facts_to_phrases = pickle.load(open(
            'output/{}_{}_graph_facts_to_phrases_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type,
                                                                           self.extraction_type, self.version), 'rb'))

        self.docs_to_facts_mat = pickle.load(
            open(
                'output/{}_{}_graph_doc_to_facts_csr_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type,
                                                                               self.extraction_type, self.version),
                'rb'))  # (num docs, num facts)
        self.facts_to_phrases_mat = pickle.load(open(
            'output/{}_{}_graph_facts_to_phrases_csr_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type,
                                                                               self.extraction_type, self.version),
            'rb'))  # (num facts, num phrases)

        self.doc_to_phrases_mat = self.docs_to_facts_mat.dot(self.facts_to_phrases_mat)
        self.doc_to_phrases_mat[self.doc_to_phrases_mat.nonzero()] = 1
        self.phrase_to_num_doc = self.doc_to_phrases_mat.sum(0).T
        ## 新增 phrase_to_doc_mat  可以从phrase id 取到所有有这个phrase的文档id
        self.phrase_to_doc_mat = self.doc_to_phrases_mat.T
        ## 新增 end

        graph_file_path = 'output/{}_{}_graph_mean_{}_thresh_{}_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type,
                                                                                          str(self.sim_threshold), self.phrase_type,
                                                                                          self.extraction_type,
                                                                                          self.graph_creating_retriever_name_processed,
                                                                                          self.version)
        if os.path.isfile(graph_file_path):
            self.graph_plus = pickle.load(open(graph_file_path, 'rb'))  # (phrase1 id, phrase2 id) -> the number of occurrences
        else:
            self.logger.exception('Graph file not found: ' + graph_file_path)

    def get_phrases_in_doc_str(self, doc: str):
        # find doc id from self.dataset_df
        try:
            if self.corpus_name == '2wikimultihopqa':
                 doc_id = self.dataset_df[self.dataset_df.title == doc].index[0]
            else:
                doc_id = self.dataset_df[self.dataset_df.paragraph == doc].index[0]
            phrase_ids = self.doc_to_phrases_mat[[doc_id], :].nonzero()[1].tolist()
            return [self.phrases[phrase_id] for phrase_id in phrase_ids]
        except:
            return []

    def build_graph(self):

        edges = set()

        new_graph_plus = {}
        self.kg_adj_list = defaultdict(dict)
        self.kg_inverse_adj_list = defaultdict(dict)

        for edge, weight in tqdm(self.graph_plus.items(), total=len(self.graph_plus), desc='Building Graph'):
            edge1 = edge[0]
            edge2 = edge[1]

            if (edge1, edge2) not in edges and edge1 != edge2:
                new_graph_plus[(edge1, edge2)] = self.graph_plus[(edge[0], edge[1])]
                edges.add((edge1, edge2))
                self.kg_adj_list[edge1][edge2] = self.graph_plus[(edge[0], edge[1])]
                self.kg_inverse_adj_list[edge2][edge1] = self.graph_plus[(edge[0], edge[1])]

        self.graph_plus = new_graph_plus

        edges = list(edges)

        n_vertices = len(self.kb_node_phrase_to_id)
        self.g = ig.Graph(n_vertices, edges)

        self.g.es['weight'] = [self.graph_plus[(v1, v3)] for v1, v3 in edges]
        self.logger.info(f'Graph built: num vertices: {n_vertices}, num_edges: {len(edges)}')

    def load_node_vectors(self):
        encoded_string_path = 'data/lm_vectors/{}_mean/encoded_strings.txt'.format(self.linking_retriever_name_processed)
        if os.path.isfile(encoded_string_path):
            self.load_node_vectors_from_string_encoding_cache(encoded_string_path)   # 拿到所有的实体节点向量 保存在 self.kb_node_phrase_embeddings
        else:  # use another way to load node vectors
            if self.linking_retriever_name == 'colbertv2':
                return
            kb_node_phrase_embeddings_path = 'data/lm_vectors/{}_mean/{}_kb_node_phrase_embeddings.p'.format(self.linking_retriever_name_processed, self.corpus_name)
            if os.path.isfile(kb_node_phrase_embeddings_path):
                self.kb_node_phrase_embeddings = pickle.load(open(kb_node_phrase_embeddings_path, 'rb'))
                if len(self.kb_node_phrase_embeddings.shape) == 3:
                    self.kb_node_phrase_embeddings = np.squeeze(self.kb_node_phrase_embeddings, axis=1)
                self.logger.info('Loaded phrase embeddings from: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))
            else:
                self.kb_node_phrase_embeddings = self.embed_model.encode_text(self.phrases.tolist(), return_cpu=True, return_numpy=True, norm=True)
                pickle.dump(self.kb_node_phrase_embeddings, open(kb_node_phrase_embeddings_path, 'wb'))
                self.logger.info('Saved phrase embeddings to: ' + kb_node_phrase_embeddings_path + ', shape: ' + str(self.kb_node_phrase_embeddings.shape))

    def load_node_vectors_from_string_encoding_cache(self, string_file_path):
        self.logger.info('Loading node vectors from: ' + string_file_path)
        kb_vectors = []
        self.strings = open(string_file_path, 'r').readlines()
        for i in range(len(glob('data/lm_vectors/{}_mean/vecs_*'.format(self.linking_retriever_name_processed)))):
            kb_vectors.append(
                torch.Tensor(pickle.load(
                    open('data/lm_vectors/{}_mean/vecs_{}.p'.format(self.linking_retriever_name_processed, i), 'rb'))))   # vec_nums * emb_dim
        kb_mat = torch.cat(kb_vectors)  # a matrix of phrase vectors   # 存放所有实体向量的mat
        self.strings = [s.strip() for s in self.strings]
        self.string_to_id = {string: i for i, string in enumerate(self.strings)}
        kb_mat = kb_mat.T.divide(torch.linalg.norm(kb_mat, dim=1)).T    # L2 norm
        kb_mat = kb_mat.to('cuda')
        kb_only_indices = []
        num_non_vector_phrases = 0
        for i in range(len(self.kb_node_phrase_to_id)):
            phrase = self.phrases[i]
            if phrase not in self.string_to_id:
                num_non_vector_phrases += 1

            phrase_id = self.string_to_id.get(phrase, 0)
            kb_only_indices.append(phrase_id)
        self.kb_node_phrase_embeddings = kb_mat[kb_only_indices]  # a matrix of phrase vectors
        self.kb_node_phrase_embeddings = self.kb_node_phrase_embeddings.cpu().numpy()
        self.logger.info('{} phrases did not have vectors.'.format(num_non_vector_phrases))

    def get_dpr_doc_embedding(self):
        cache_filename = 'data/lm_vectors/{}_mean/{}_doc_embeddings.p'.format(self.linking_retriever_name_processed, self.corpus_name)
        if os.path.exists(cache_filename):
            self.doc_embedding_mat = pickle.load(open(cache_filename, 'rb'))
            self.logger.info(f'Loaded doc embeddings from {cache_filename}, shape: {self.doc_embedding_mat.shape}')
        else:
            self.doc_embeddings = []
            self.doc_embedding_mat = self.embed_model.encode(self.dataset_df['paragraph'].tolist(), normalize_embeddings=True, device='cuda', batch_size=16) # 原本64
            pickle.dump(self.doc_embedding_mat, open(cache_filename, 'wb'))
            self.logger.info(f'Saved doc embeddings to {cache_filename}, shape: {self.doc_embedding_mat.shape}')


    def get_colbert_max_score(self, query):
        queries_ = [query]
        encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)
        encoded_doc = self.phrase_searcher.checkpoint.docFromText(queries_).float()
        max_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

        return max_score

    def get_colbert_real_score(self, query, doc):
        queries_ = [query]
        encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)

        docs_ = [doc]
        encoded_doc = self.phrase_searcher.checkpoint.docFromText(docs_).float()

        real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

        return real_score

    def link_node_by_colbertv2(self, query_ner_list):
        phrase_ids = []
        max_scores = []

        for query in query_ner_list:
            queries = Queries(path=None, data={0: query})

            queries_ = [query]
            encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)

            max_score = self.get_colbert_max_score(query)

            ranking = self.phrase_searcher.search_all(queries, k=1)
            for phrase_id, rank, score in ranking.data[0]:
                phrase = self.phrases[phrase_id]
                phrases_ = [phrase]
                encoded_doc = self.phrase_searcher.checkpoint.docFromText(phrases_).float()
                real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

                phrase_ids.append(phrase_id)
                max_scores.append(real_score / max_score)

        # create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        top_phrase_vec = np.zeros(len(self.phrases))

        for phrase_id in phrase_ids:
            if self.node_specificity:
                if self.phrase_to_num_doc[phrase_id] == 0:
                    weight = 1
                else:
                    weight = 1 / self.phrase_to_num_doc[phrase_id]
                top_phrase_vec[phrase_id] = weight
            else:
                top_phrase_vec[phrase_id] = 1.0

        return top_phrase_vec, {(query, self.phrases[phrase_id]): max_score for phrase_id, max_score, query in zip(phrase_ids, max_scores, query_ner_list)}

    def link_node_by_dpr(self, query_ner_list: list):
        """
        Get the most similar phrases (as vector) in the KG given the named entities
        :param query_ner_list:
        :return:
        """
        query_ner_embeddings = self.embed_model.encode_text(query_ner_list, return_cpu=True, return_numpy=True, norm=True)

        # Get Closest Entity Nodes
        prob_vectors = np.dot(query_ner_embeddings, self.kb_node_phrase_embeddings.T)  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)

        linked_phrase_ids = []
        max_scores = []

        for prob_vector in prob_vectors:
            # 源代码
            phrase_id = np.argmax(prob_vector)  # the phrase with the highest similarity  只取了相似性最大的一个id
            linked_phrase_ids.append(phrase_id)
            max_scores.append(prob_vector[phrase_id])
            # 源代码 end
            
            # ## 更改代码 ====
            # # 获取 top n 个最大索引
            # n = 2
            # top_n_indices = np.argsort(prob_vector)[-n:][::-1]
            # for phrase_id in top_n_indices:
            #     linked_phrase_ids.append(phrase_id)
            #     max_scores.append(prob_vector[phrase_id])
            # # 更改代码 end ===


        # create a vector (num_phrase) with 1s at the indices of the linked phrases and 0s elsewhere
        # if node_specificity is True, it's not one-hot but a weight
        all_phrase_weights = np.zeros(len(self.phrases))

        for phrase_id in linked_phrase_ids:
            if self.node_specificity:
                if self.phrase_to_num_doc[phrase_id] == 0:  # just in case the phrase is not recorded in any documents
                    weight = 1
                else:  # the more frequent the phrase, the less weight it gets
                    weight = 1 / self.phrase_to_num_doc[phrase_id]

                all_phrase_weights[phrase_id] = weight
            else:
                all_phrase_weights[phrase_id] = 1.0

        linking_score_map = {(query_phrase, self.phrases[linked_phrase_id]): max_score
                             for linked_phrase_id, max_score, query_phrase in zip(linked_phrase_ids, max_scores, query_ner_list)}
        return all_phrase_weights, linking_score_map
