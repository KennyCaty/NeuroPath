import json
import logging
import os
import _pickle as pickle
from glob import glob

import numpy as np
import pandas as pd
import torch

from src.langchain_util import init_langchain_model
from src.lm_wrapper.util import init_embedding_model
from src.query_ner_vtp_parallel import  ner_vtp_extraction
from src.processing import processing_phrases
from langchain_core.messages import HumanMessage, SystemMessage

import time

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'



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

    def __init__(self, corpus_name='hotpotqa',
                 index_llm='openai', index_llm_model='gpt-4o-mini',
                 rag_llm='openai', rag_llm_model='gpt-4o-mini',
                 graph_creating_retriever_name='facebook/contriever', extraction_type='ner', graph_type='facts', node_specificity=True,
                 dpr_only=False, graph_alg='kg_path', corpus_path=None,
                 linking_retriever_name=None, max_hop=2):

        self.max_hop = max_hop
        self.corpus_name = corpus_name
        self.index_llm_model = index_llm_model
        self.index_llm_model_processed = index_llm_model.replace('/', '_')
        # rag_llm drives retrieval-time path tracking and query NER
        self.client = init_langchain_model(rag_llm, rag_llm_model, role='rag')
        assert graph_creating_retriever_name
        if linking_retriever_name is None:
            linking_retriever_name = graph_creating_retriever_name
        self.graph_creating_retriever_name = graph_creating_retriever_name
        self.graph_creating_retriever_name_processed = graph_creating_retriever_name.replace('/', '_').replace('.', '')
        self.linking_retriever_name = linking_retriever_name
        self.linking_retriever_name_processed = linking_retriever_name.replace('/', '_').replace('.', '')

        # extraction_type encodes phrase-extraction scheme + indexing LLM, so that
        # graph artifacts are uniquely keyed on (extraction scheme, index_llm_model).
        self.extraction_type = f'{extraction_type}_{self.index_llm_model_processed}'
        self.graph_type = graph_type
        self.phrase_type = 'ents_only_lower_preprocess'
        self.node_specificity = node_specificity

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
        self.fact_json = json.load(open(
            'output/{}_{}_graph_clean_facts_chatgpt_openIE.{}_{}.{}.subset.json'.format(
                corpus_name, graph_type, self.phrase_type, self.extraction_type, self.version), 'r'))

        # Loading Important Corpus Files
        if not self.dpr_only:
            self.load_index_files()

            # Loading Node Embeddings
            self.load_node_vectors()
        else:
            self.load_corpus()

        if dpr_only or graph_alg == 'kg_path':
            # Loading Doc Embeddings
            self.get_dpr_doc_embedding()

        self.statistics = {}
        self.ensembling_debug = []

        # ===== Build phrase faiss index for pseudo-coreference resolution =====
        import faiss
        faiss.normalize_L2(self.kb_node_phrase_embeddings)
        resource = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(self.kb_node_phrase_embeddings.shape[1])
        gpu_index = faiss.index_cpu_to_gpu(resource, 0, index)
        gpu_index.add(self.kb_node_phrase_embeddings)
        k = 5  # retrieve top-5 most similar (excluding self)

        distances, indices = gpu_index.search(self.kb_node_phrase_embeddings, k)

        # Pseudo-Coreference Resolution: keep neighbors with similarity >= 0.8
        self.coreference_resolution = []
        for i in range(len(self.kb_node_phrase_embeddings)):
            valid_indices = [idx for idx, dist in zip(indices[i], distances[i]) if dist >= 0.8 and idx != i]
            self.coreference_resolution.append(valid_indices)
        # ======================================================================



   
    def llm_path_track(self, path_prompt, query, one_shot=False):
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
        if one_shot == True:
            sys_prompt += """
# Example
Query: Which film has the director who was born later, El Extrano Viaje or Love In Pawn?
Paths:
0: El Extrano Viaje->released in->1964;
1: El Extrano Viaje->directed by->Fernando Fernan Gomez; <expandable>: [Fernando Fernan Gomez]
2: El Extrano Viaje->starring->Jose Isbert; <expandable>: [Jose Isbert]
3: Love in Pawn->released in->1953;
4: Love In Pawn->directed by->Charles Saunders; <expandable>: [Charles Saunders]
{Output:
"current_chain": "The director of El Extrano Viaje is Fernando Fernan Gomez. And the director of Love In
Pawn is Charles Saunders",
"valid_ids": [1,4],
"expansion_requirements": "Find when were Fernando Fernan Gomez and Charles Saunders born.",
"need_expand_ids": [1,4],
"continue": 1
}
# Example End
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
        
        # only prune newly expanded paths
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
        
        
        path_dict['paths'] = [path_dict['paths'][i] for i in indices]
        path_dict['link_phrases'] = [path_dict['link_phrases'][i] for i in indices]
        path_dict['visited_docs'] = [path_dict['visited_docs'][i] for i in indices]
        return path_dict
    

    def Expand_by_llm(self, path_dict, query_embedding, query, vtp=None, one_shot=False): 
        total_tokens = 0
        # --- First round ---
        if path_dict['paths'] == []:
            new_link_phrases = []
            expand_phrase_ids = self.expand_seed(path_dict['link_phrases'])
            for link_phrase_id in expand_phrase_ids:
                if self.phrases[link_phrase_id] == 'NaN' or self.phrases[link_phrase_id] == 'Null' or self.phrases[link_phrase_id] == '':  # empty phrase, skip
                    continue
                # docs containing this phrase
                doc_row = self.phrase_to_doc_mat[[link_phrase_id]].toarray()[0]
                non_zero_doc_ids = np.nonzero(doc_row)[0].tolist()
                # collect sub-paths containing the phrase
                for doc_id in non_zero_doc_ids:
                    target_facts_ids_list = self.docs_to_facts_mat[[doc_id]].toarray()[0]
                    non_zero_target_facts_ids = np.nonzero(target_facts_ids_list)[0].tolist()
                    
                    for facts_id in non_zero_target_facts_ids:
                        if self.facts_to_phrases_mat[facts_id, link_phrase_id] != 0:
                            triple_json = self.fact_json[facts_id]
                            triple_semantics =  triple_json['head'] + '->' + triple_json['relation'] + '->' + triple_json['tail'] + '; '
                            if triple_semantics in path_dict['paths']:
                                continue
                            head_id = self.kb_node_phrase_to_id[triple_json['head']]
                            tail_id = self.kb_node_phrase_to_id[triple_json['tail']]
                            # the other phrase connected in the sub-path
                            if head_id==link_phrase_id:
                                new_link_phrase_id = tail_id
                            else:
                                new_link_phrase_id = head_id
                            new_link_phrases.append(new_link_phrase_id)
                            path_dict['paths'].append(triple_semantics)
                            path_dict['visited_docs'].append([doc_id])
                            
            path_dict['link_phrases'] = new_link_phrases
            
            # similarity-based pruning
            path_dict = self.pruning(0, vtp, path_dict)
            
            path_prompt = ''
            for i in range(len(path_dict['paths'])):
                path_prompt += f"{i}: " + path_dict['paths'][i] + ' <expandable>: [' + self.phrases[path_dict['link_phrases'][i]] + ']' + '\n'
            
            response_json, tokens, llm_time = self.llm_path_track(path_prompt, query, one_shot)
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
                
                path_dict['current_chain'] = response_json['current_chain']
                path_dict['expansion_requirements'] = response_json['expansion_requirements']

            except Exception as e:
                print(e, "response parsing error")
                print(path_dict)
                print(response_json)
                path_dict['valid_paths'] = []
                path_dict['visited_docs'] = []
                path_dict['valid_visited_docs'] = []
                path_dict['need_expand'] = []
                path_dict['paths'] = []
                path_dict['current_chain'] = ''
                path_dict['expansion_requirements'] = ''

                flag = 0
            return path_dict, flag, total_tokens, llm_time
           
           
           
        # --- Subsequent rounds ---
        else:
            # number of valid paths from the previous round
            old_path_len = len(path_dict['paths'])
            
            new_link_phrases = []
            new_paths = []
            new_visited_docs = []
            locked_path = []
            for i in range(len(path_dict['paths'])):
                path = path_dict['paths'][i]
                if path not in path_dict['need_expand']:
                    locked_path.append(path)
            for i in range(len(path_dict['paths'])):  # expand each expandable path
                if path_dict['paths'][i] not in locked_path:
                    path = path_dict['paths'][i]
                    seed_phrase_id = path_dict['link_phrases'][i]
                    # expand similar phrases (coreference resolution)
                    expand_phrase_ids = self.expand_seed([seed_phrase_id])
                    visited_docs = path_dict['visited_docs'][i]
                    
                    for link_phrase_id in expand_phrase_ids:
                        if self.phrases[link_phrase_id] == 'NaN' or self.phrases[link_phrase_id] == 'Null' or self.phrases[link_phrase_id] == '':  # empty phrase, skip
                            continue
                        doc_row = self.phrase_to_doc_mat[[link_phrase_id]].toarray()[0]
                        non_zero_doc_ids = np.nonzero(doc_row)[0].tolist()
                        # docs linked to link_phrase (unvisited)
                        for doc_id in non_zero_doc_ids:
                            target_facts_ids_list = self.docs_to_facts_mat[[doc_id]].toarray()[0]
                            non_zero_target_facts_ids = np.nonzero(target_facts_ids_list)[0].tolist()
                            for facts_id in non_zero_target_facts_ids:
                                if self.facts_to_phrases_mat[facts_id, link_phrase_id] != 0:
                                    triple_json = self.fact_json[facts_id]
                                    triple_semantics =  triple_json['head'] + '->' + triple_json['relation'] + '->' + triple_json['tail'] + '; '
                                    head_id = self.kb_node_phrase_to_id[triple_json['head']]
                                    tail_id = self.kb_node_phrase_to_id[triple_json['tail']]
                                    if triple_semantics in path_dict['paths'] or triple_semantics in path_dict['need_expand']:
                                        continue
                                    if head_id==link_phrase_id:
                                        new_link_phrase_id = tail_id
                                    else:
                                        new_link_phrase_id = head_id
                                    new_link_phrases.append(new_link_phrase_id)
                                    path_i = path
                                    path_i += triple_semantics
                                    new_paths.append(path_i)
                                    new_visited_docs.append(visited_docs + [doc_id])
            path_dict['link_phrases'].extend(new_link_phrases)
            path_dict['paths'].extend(new_paths)
            path_dict['visited_docs'].extend(new_visited_docs)
            
            # similarity-based pruning
            path_dict = self.pruning(old_path_len, vtp, path_dict)
            
            path_prompt = ''
            for i in range(len(path_dict['paths'])):
                if path_dict['paths'][i] in locked_path:  # lock paths that do not need expansion
                    path_prompt += f"{i}: " + path_dict['paths'][i] + '\n'
                elif path_dict['paths'][i] in path_dict['need_expand']:  # lock already-expanded paths
                    path_prompt += f"{i}: " + path_dict['paths'][i] + '\n'
                else:
                    path_prompt += f"{i}: " + path_dict['paths'][i] + ' <expandable>: [' + self.phrases[path_dict['link_phrases'][i]] + ']' + '\n'
                    
            
            response_json, tokens, llm_time = self.llm_path_track(path_prompt, query, one_shot)
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
                path_dict['paths'] = [path_dict['paths'][i] for i in new_path_ids_list]
                path_dict['link_phrases'] = [path_dict['link_phrases'][i] for i in new_path_ids_list]
                path_dict['visited_docs'] = [path_dict['visited_docs'][i] for i in new_path_ids_list]         
                
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

        
        
    def get_path_filtered_docs(self, query_embedding, seed_phrase_ids, query, one_shot):
        unique_phrase_ids_set = set(seed_phrase_ids)
        # init state
        path_dict = {}
        path_dict['paths'] = []
        path_dict['link_phrases'] = list(unique_phrase_ids_set)
        path_dict['visited_docs'] = []  # visited doc ids per path
        path_dict['need_expand_ids'] = []
        flag = 1
        max_hop = self.max_hop
        count = 0
        total_tokens = 0
        cur_vtp = query
        
        all_thought = ''
        llm_time_for_one_q = 0
        while count < max_hop and flag==1:
            count += 1
            path_dict, flag, tokens, llm_time = self.Expand_by_llm(path_dict, query_embedding, query, cur_vtp, one_shot)
            llm_time_for_one_q += llm_time
            total_tokens += tokens
            all_thought = path_dict['current_chain'] + path_dict['expansion_requirements']
            cur_vtp =  path_dict['expansion_requirements']

            if flag==0 and path_dict['valid_paths'] != []:
                print("The LLM determines the path.")
            elif path_dict['valid_paths'] == [] and path_dict['need_expand'] == []:
                print("The LLM failed to find a path.")
                flag = 0
            
        
        # collect candidate doc ids
        candidate_docs = []
        candidate_paths = path_dict['valid_paths']
        visited_docs = path_dict['valid_visited_docs']
        for each_path_docs in visited_docs:
            for doc in each_path_docs:
                if doc not in candidate_docs:
                    candidate_docs.append(doc)

        return candidate_docs, candidate_paths, total_tokens, all_thought, llm_time_for_one_q
       
 
    
    def rank_docs(self, query: str, top_k, one_shot=False):
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

            if len(query_ner_list) > 0:  # use Path
                query_ner_embeddings = self.embed_model.encode(query_ner_list, normalize_embeddings=True, device='cuda', show_progress_bar=False)
                # Get Closest Entity Nodes
                prob_vectors = np.dot(query_ner_embeddings, self.kb_node_phrase_embeddings.T)  # (num_ner, dim) x (num_phrases, dim).T -> (num_ner, num_phrases)
                linked_phrase_ids = []  # Seed Phrases

                for prob_vector in prob_vectors:
                    phrase_id = np.argmax(prob_vector)  # the phrase with the highest similarity
                    linked_phrase_ids.append(phrase_id)
                
                
                # get re-ranked doc ids
                candidate_doc_ids, candidate_paths, total_tokens, all_thought, llm_time_for_one_q = self.get_path_filtered_docs(None, linked_phrase_ids, query, one_shot)
                
                query_embedding = self.embed_model.encode([query+all_thought], normalize_embeddings=True, device='cuda', show_progress_bar=False)
                
                query_doc_scores = np.dot(self.doc_embedding_mat, query_embedding.T)
                query_doc_scores = query_doc_scores.T[0]

                doc_ids = []
                doc_scores = []
                # all-doc similarity scores
                doc_prob = query_doc_scores
                # sorted ids and scores
                sorted_doc_ids = np.argsort(doc_prob)[::-1]
                sorted_scores = doc_prob[sorted_doc_ids]
                
                for t in candidate_doc_ids:
                    doc_ids.append(t)
                    doc_scores.append(1)  # placeholder score
                # pad with the remaining top docs to keep top-k recall meaningful
                for idx in range(100):
                    if sorted_doc_ids[idx] not in doc_ids:
                        doc_ids.append(sorted_doc_ids[idx])
                        doc_scores.append(sorted_scores[idx])
            else:  # fallback to pure similarity retrieval
                query_embedding = self.embed_model.encode([query], normalize_embeddings=True, device='cuda', show_progress_bar=False)
                query_doc_scores = np.dot(self.doc_embedding_mat, query_embedding.T)
                query_doc_scores = query_doc_scores.T[0]


                doc_ids = []
                doc_scores = []
                doc_prob = query_doc_scores
                sorted_doc_ids = np.argsort(doc_prob)[::-1]
                sorted_scores = doc_prob[sorted_doc_ids]
                
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

    def load_corpus(self):
        if self.corpus_path is None:
            self.corpus_path = 'data/{}_corpus.json'.format(self.corpus_name)
        assert os.path.isfile(self.corpus_path), 'Corpus file not found'
        self.corpus = json.load(open(self.corpus_path, 'r'))
        self.dataset_df = pd.DataFrame()
        self.dataset_df['paragraph'] = [p['title'] + '\n' + p['text'] for p in self.corpus]

    def load_index_files(self):
        index_file_pattern = 'output/openie_{}_results_{}_{}_*.json'.format(self.corpus_name, self.extraction_type, self.index_llm_model_processed)
        possible_files = glob(index_file_pattern)
        if len(possible_files) == 0:
            self.logger.critical(f'No extraction files found: {index_file_pattern} ; please check if working directory is correct or if the extraction has been done.')
            return
        max_samples = np.max(
            [int(file.split('{}_'.format(self.index_llm_model_processed))[1].split('.json')[0]) for file in possible_files])
        extracted_file = json.load(open(
            'output/openie_{}_results_{}_{}_{}.json'.format(self.corpus_name, self.extraction_type, self.index_llm_model_processed, max_samples),
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

        self.kb_node_phrase_to_id = pickle.load(open(
            'output/{}_{}_graph_phrase_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type,
                                                                      self.extraction_type, self.version), 'rb'))
        self.lose_fact_dict = pickle.load(open(
            'output/{}_{}_graph_fact_dict_{}_{}.{}.subset.p'.format(self.corpus_name, self.graph_type, self.phrase_type,
                                                                    self.extraction_type, self.version), 'rb'))

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
        # phrase_to_doc_mat: given a phrase id, fetch all doc ids containing it
        self.phrase_to_doc_mat = self.doc_to_phrases_mat.T

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

    def load_node_vectors(self):
        encoded_string_path = 'data/lm_vectors/{}_mean/encoded_strings.txt'.format(self.linking_retriever_name_processed)
        if os.path.isfile(encoded_string_path):
            self.load_node_vectors_from_string_encoding_cache(encoded_string_path)   # loads all KB node embeddings into self.kb_node_phrase_embeddings
        else:  # use another way to load node vectors
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
        kb_mat = torch.cat(kb_vectors)  # matrix of phrase vectors
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
            self.doc_embedding_mat = self.embed_model.encode(self.dataset_df['paragraph'].tolist(), normalize_embeddings=True, device='cuda', batch_size=16)
            pickle.dump(self.doc_embedding_mat, open(cache_filename, 'wb'))
            self.logger.info(f'Saved doc embeddings to {cache_filename}, shape: {self.doc_embedding_mat.shape}')
