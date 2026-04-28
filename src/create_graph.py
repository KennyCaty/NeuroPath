import pandas as pd
import numpy as np
from scipy.sparse import csr_array
from processing import *
from glob import glob

import os
import json
from tqdm import tqdm
import pickle
import argparse

os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'


def create_graph(dataset: str, extraction_type: str, index_llm_model: str, retriever_name: str, processed_retriever_name: str,
                 create_graph_flag: bool = False):
    version = 'v3'
    possible_files = glob('output/openie_{}_results_{}_{}_*.json'.format(dataset, extraction_type, index_llm_model))
    max_samples = np.max([int(file.split('{}_'.format(index_llm_model))[1].split('.json')[0]) for file in possible_files])
    extracted_file = json.load(open('output/openie_{}_results_{}_{}_{}.json'.format(dataset, extraction_type, index_llm_model, max_samples), 'r'))

    extracted_triples = extracted_file['docs']
    extraction_type = extraction_type + '_' + index_llm_model
    phrase_type = 'ents_only_lower_preprocess'
    graph_type = 'facts'

    phrases = []
    entities = []
    relations = {}
    incorrectly_formatted_triples = []
    triples_wo_ner_entity = []
    triple_tuples = []
    full_neighborhoods = {}
    correct_wiki_format = 0

    for i, row in tqdm(enumerate(extracted_triples), total=len(extracted_triples)):
        ner_entities = [processing_phrases(p) for p in row['extracted_entities']]

        triples = row['extracted_triples']

        clean_triples = []

        for triple in triples:
            triple = [str(s) for s in triple]

            if len(triple) > 1:
                if len(triple) != 3:
                    incorrectly_formatted_triples.append(triple)
                else:
                    clean_triple = [processing_phrases(p) for p in triple]

                    clean_triples.append(clean_triple)
                    phrases.extend(clean_triple)

                    head_ent = clean_triple[0]
                    tail_ent = clean_triple[2]

                    if head_ent not in ner_entities and tail_ent not in ner_entities:
                        triples_wo_ner_entity.append(triple)

                    relations[(head_ent, tail_ent)] = clean_triple[1]

                    raw_head_ent = triple[0]
                    raw_tail_ent = triple[2]

                    entity_neighborhood = full_neighborhoods.get(raw_head_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_head_ent] = entity_neighborhood

                    entity_neighborhood = full_neighborhoods.get(raw_tail_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_tail_ent] = entity_neighborhood

                    for triple_entity in [clean_triple[0], clean_triple[2]]:
                        entities.append(triple_entity)

        triple_tuples.append(clean_triples)

    print('Correct Wiki Format: {} out of {}'.format(correct_wiki_format, len(extracted_triples)))

    unique_phrases = list(np.unique(entities))
    unique_relations = np.unique(list(relations.values()) + ['equivalent'])

    if create_graph_flag:
        print('Creating Graph')
        kb_phrase_dict = {p: i for i, p in enumerate(unique_phrases)}

        lose_facts = []
        for triples in triple_tuples:
            lose_facts.extend([tuple(t) for t in triples])

        lose_fact_dict = {f: i for i, f in enumerate(lose_facts)}
        fact_json = [{'idx': i, 'head': t[0], 'relation': t[1], 'tail': t[2]} for i, t in enumerate(lose_facts)]

        json.dump(fact_json, open('output/{}_{}_graph_clean_facts_chatgpt_openIE.{}_{}.{}.subset.json'.format(dataset, graph_type, phrase_type, extraction_type, version), 'w'))
        pickle.dump(kb_phrase_dict, open('output/{}_{}_graph_phrase_dict_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))
        pickle.dump(lose_fact_dict, open('output/{}_{}_graph_fact_dict_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))

        docs_to_facts = {}  # (num docs, num facts)
        facts_to_phrases = {}  # (num facts, num phrases)

        for doc_id, triples in tqdm(enumerate(triple_tuples), total=len(triple_tuples)):
            for triple in triples:
                triple = tuple(triple)
                if len(triple) != 3:
                    continue
                fact_id = lose_fact_dict[triple]
                docs_to_facts[(doc_id, fact_id)] = 1
                for phrase in (triple[0], triple[2]):
                    phrase_id = kb_phrase_dict[phrase]
                    facts_to_phrases[(fact_id, phrase_id)] = 1

        pickle.dump(docs_to_facts, open('output/{}_{}_graph_doc_to_facts_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))
        pickle.dump(facts_to_phrases, open('output/{}_{}_graph_facts_to_phrases_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))

        docs_to_facts_mat = csr_array(([int(v) for v in docs_to_facts.values()],
                                       ([int(e[0]) for e in docs_to_facts.keys()], [int(e[1]) for e in docs_to_facts.keys()])),
                                      shape=(len(triple_tuples), len(lose_facts)))
        facts_to_phrases_mat = csr_array(([int(v) for v in facts_to_phrases.values()],
                                          ([e[0] for e in facts_to_phrases.keys()], [e[1] for e in facts_to_phrases.keys()])),
                                         shape=(len(lose_facts), len(unique_phrases)))

        pickle.dump(docs_to_facts_mat, open('output/{}_{}_graph_doc_to_facts_csr_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))
        pickle.dump(facts_to_phrases_mat, open('output/{}_{}_graph_facts_to_phrases_csr_{}_{}.{}.subset.p'.format(dataset, graph_type, phrase_type, extraction_type, version), 'wb'))

        stat_df = [('Total Phrases', len(phrases)),
                   ('Unique Phrases', len(unique_phrases)),
                   ('Number of Individual Triples', len(lose_facts)),
                   ('Number of Incorrectly Formatted Triples (ChatGPT Error)', len(incorrectly_formatted_triples)),
                   ('Number of Triples w/o NER Entities (ChatGPT Error)', len(triples_wo_ner_entity)),
                   ('Number of Unique Individual Triples', len(lose_fact_dict)),
                   ('Number of Entities', len(entities)),
                   ('Number of Relations', len(relations)),
                   ('Number of Unique Entities', len(np.unique(entities))),
                   ('Number of Unique Relations', len(unique_relations))]

        print(pd.DataFrame(stat_df).set_index(0))

        print('Saving Graph')


if __name__ == '__main__':
    # Get the first argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--index_llm_model', type=str)
    parser.add_argument('--create_graph', action='store_true')
    parser.add_argument('--extraction_type', type=str)

    args = parser.parse_args()
    dataset = args.dataset
    retriever_name = args.model_name
    processed_retriever_name = retriever_name.replace('/', '_').replace('.', '')
    index_llm_model = args.index_llm_model.replace('/', '_')
    create_graph_flag = args.create_graph
    extraction_type = args.extraction_type

    create_graph(dataset, extraction_type, index_llm_model, retriever_name, processed_retriever_name, create_graph_flag)
