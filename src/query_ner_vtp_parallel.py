import sys
from functools import partial

sys.path.append('.')

from src.processing import extract_json_dict
from langchain_community.chat_models import ChatLlamaCpp
from langchain_ollama import ChatOllama

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from tqdm import tqdm

from src.langchain_util import init_langchain_model





sys_prompt = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

# Example 1
Question: Who wrote the book that was published earlier, "The Great Gatsby" or "To Kill a Mockingbird"?
Output :
{
    "named_entities": ["The Great Gatsby", "To Kill a Mockingbird"]
}
# Example 2
Question : Are the cities of Paris and Berlin located in the same country?
Output :
{
    "named_entities": ["Paris", "Berlin"]
}
"""

input_frame = """Question: {}
Output:
"""

def ner_vtp_extraction(client, text: str):
    query_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage(sys_prompt),
                                                          HumanMessage(input_frame.format(text))])
    query_ner_messages = query_ner_prompts.format_prompt()

    json_mode = False
    try:
        if isinstance(client, ChatOpenAI):  # JSON mode
            chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=1000, stop=['\n\n'], response_format={"type": "json_object"})
            response_content = chat_completion.content
            total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
            json_mode = True
        elif isinstance(client, ChatOllama) or isinstance(client, ChatLlamaCpp):
            response_content = client.invoke(query_ner_messages.to_messages())
            response_content = extract_json_dict(response_content)
            total_tokens = len(response_content.split())
        else:  # no JSON mode
            chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=1000, stop=['\n\n'])
            response_content = chat_completion.content
            response_content = extract_json_dict(response_content)
            total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
    except Exception as e:
        print(e)
        response_content = {'named_entities': []}
        total_tokens = 0
    if not json_mode:
        try:
            assert 'named_entities' in response_content
            response_content = str(response_content)
        except Exception as e:
            print('Query NER exception', e)
            response_content = {'named_entities': []}

    return response_content, total_tokens


def run_ner_vtp_on_texts(texts, llm='openai', model_name='gpt-4o-mini'):
    client = init_langchain_model(llm, model_name, role='index')
    outputs = []
    total_cost = 0

    for text in tqdm(texts):
        out, cost = ner_vtp_extraction(client, text)
        outputs.append(out)
        total_cost += cost

    return outputs, total_cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--llm', type=str, default='openai', help="LLM provider, e.g., 'openai' or 'together'")
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Specific model name')
    parser.add_argument('--num_processes', type=int, default=8, help='Number of processes')

    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model_name
    model_name_processed = model_name.replace('/', '_')

    output_file = 'output/{}_{}_queries.named_entity_output.tsv'.format(dataset, model_name_processed)

    try:
        queries_df = pd.read_json(f'data/{dataset}.json')

        if 'hotpotqa' in dataset:
            queries_df = queries_df[['question']]
            queries_df['0'] = queries_df['question']
            queries_df['query'] = queries_df['question']
            query_name = 'query'
        else:
            query_name = 'question'

        try:
            output_df = pd.read_csv(output_file, sep='\t')
        except:
            output_df = []

        if len(queries_df) != len(output_df):
            queries = queries_df[query_name].values

            num_processes = args.num_processes

            splits = np.array_split(range(len(queries)), num_processes)

            split_args = []

            for split in splits:
                split_args.append([queries[i] for i in split])

            partial_func = partial(run_ner_vtp_on_texts, llm=args.llm, model_name=model_name)
            if num_processes == 1:
                outputs = [partial_func(split_args[0])]
            else:
                with Pool(processes=num_processes) as pool:
                    outputs = pool.map(partial_func, split_args)

            chatgpt_total_tokens = 0

            query_triples = []

            for output in outputs:
                query_triples.extend(output[0])
                chatgpt_total_tokens += output[1]

            print("Tokens: ", chatgpt_total_tokens)

            queries_df['triples'] = query_triples
            queries_df.to_csv(output_file, sep='\t')
            print('Query NER saved to', output_file)
        else:
            print('Query NER already saved to', output_file)
    except Exception as e:
        print('No queries will be processed for later retrieval.', e)
