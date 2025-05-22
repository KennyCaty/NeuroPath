from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate

## NER + KG Prompt
sys_prompt = """Your task is to extract named entities from the given passage and construct an KG(Knowledge Graph) from the passage and the entities you extracted.
Requirements:
The entity type can be [organization, person, object, location, time, event, term], etc. And the KG should describe the information contained in the text as detailed as possible.
The format of a KG (Knowledge Graph) triple is ["head node", "relation", "tail node"], and each part must have a value.
Coreference and Pronoun Resolution:
Specific names should be explicitly resolved to maintain clarity.
Respond with a JSON Object.

# Example Begin:
Passage:
```
Teutberga
Teutberga( died 11 November 875) was a queen of Lotharingia by marriage to Lothair II. She was a daughter of Bosonid Boso the Elder and sister of Hucbert, the lay- abbot of St. Maurice's Abbey.
```
Output:
```
{
    "named_entities": ["Teutberga", "11 November 875", "Lotharingia", "Lothair II", "Bosonid Boso the Elder", "Hucbert", "St. Maurice's Abbey"],
    "triples": [
        ["Teutberga", "died on", "11 November 875"],
        ["Teutberga", "was a queen of", "Lotharingia"],
        ["Teutberga", "married to", "Lothair II"],
        ["Teutberga", "is a daughter of", "Bosonid Boso the Elder"],
        ["Teutberga", "is a sister of", "Hucbert"],
        ["Hucbert", "is the lay-abot of", "St. Maurice's Abbey"]
    ],
}
```
# Example End
"""

input_frame = """Passage:
```
{passage}
```
Output:
"""



openie_prompt = ChatPromptTemplate.from_messages([SystemMessage(sys_prompt),
                                                    HumanMessagePromptTemplate.from_template(input_frame)])