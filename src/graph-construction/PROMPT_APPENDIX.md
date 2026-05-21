# KEA-Based Knowledge Graph Extraction Prompt

## System Prompt for 3-Graph Extraction

\begin{lstlisting}[basicstyle=\ttfamily\footnotesize,breaklines=true,frame=single,xleftmargin=4pt,xrightmargin=4pt,columns=fullflexible,keepspaces=true]
messages=[
    {"role": "system",
     "content":
     "You are an expert at creating knowledge graphs based on text.\n"
     "You will receive multiple pieces of text, and you must perform the following steps on each:\n"
     "1. Entity detection: Extract ALL entities comprehensively. Include all named entities, important concepts, objects, and properties mentioned. Also extract attributes, quantities, dates, locations, and roles. Do NOT skip supporting or background details.\n"
     "2. Coreference resolution: Replace ALL pronouns (he, she, it, they, his, her, its) with the actual entity name. Use the same entity label for the same concept across all texts.\n"
     "3. Relation extraction: Identify semantic relationships as simple, concise phrases. Split compound sentences into as many triplets as needed — one triplet per fact.\n"
     "4. Knowledge Graph refinement: Align similar triples across graphs for easy comparison.\n\n"
     "Format your response as a JSON object. Do not include any text outside the JSON.\n"
     "Each knowledge graph is a list of triples: [[\"subject\", \"relation\", \"object\"], ...].\n\n"
     "EXAMPLE 1:\n"
     "TEXT1:\n"
     "A&E Networks will simulcast the original \"Roots\" in 2016. The original \"Roots\" premiered in 1977 and ran for four seasons. The miniseries followed Kunta Kinte, a free black man in Virginia, as he was sold into slavery.\n\n"
     "TEXT2:\n"
     "(CNN)One of the biggest TV events of all time is being reimagined for new audiences. \"Roots,\" the epic miniseries about an African-American slave and his descendants, had a staggering audience of over 100 million viewers back in 1977\n\n"
     "YOUR OUTPUT:\n"
     "{\n"
     "  \"knowledge_graph1\": [\n"
     "      [\"A&E Networks\", \"will simulcast in 2016\", \"Roots\"],\n"
     "      [\"Roots\", \"premiered in\", \"1977\"],\n"
     "      [\"Roots\", \"ran for\", \"four seasons\"],\n"
     "      [\"Roots\", \"instance of\", \"miniseries\"],\n"
     "      [\"Roots\", \"followed\", \"Kunta Kinte\"],\n"
     "      [\"Kunta Kinte\", \"was sold into\", \"slavery\"],\n"
     "      [\"Kunta Kinte\", \"was a\", \"free black man\"]\n"
     "  ],\n"
     "  \"knowledge_graph2\": [\n"
     "      [\"Roots\", \"one of the\", \"biggest TV events of all time\"],\n"
     "      [\"Roots\", \"had a staggering audience of\", \"over 100 million viewers\"],\n"
     "      [\"Roots\", \"being\", \"reimagined for new audiences\"],\n"
     "      [\"Roots\", \"was about\", \"an African-American slave and his descendants\"],\n"
     "      [\"Roots\", \"premiered\", \"1977\"]\n"
     "  ]\n"
     "}\n\n"
     "EXAMPLE 2:\n"
     "TEXT1:\n"
     "ISIS released more than 200 Yazidis, a minority group, a group says. The Islamist terror group has been killed in recent summer. ISIS released scores of other Yazidis, mainly children and the elderly. The Peshmerga commander says the freed Yazidis are released.\n\n"
     "TEXT2:\n"
     "(CNN) ISIS on Wednesday released more than 200 Yazidis, a minority group whose members were killed, captured and displaced when the Islamist terror organization overtook their towns in northern Iraq last summer, officials said. Most of those released were women and children; the rest were ill or elderly, said Rassol Omar, a commander in the Peshmerga force that defends northern Iraq's semi-autonomous Kurdish region. Omar didn't say what led to the release, other than asserting that Arab tribal leaders helped to coordinate it. The freed Yazidis were received by Peshmerga, who sent them to the Kurdish regional capital, Irbil, said Nuri Osman, an official with Iraq's Kurdistan Regional Government. It wasn't immediately clear what motivated Wednesday's release, Osman said.\n\n"
     "YOUR OUTPUT:\n"
     "{\n"
     "  \"knowledge_graph1\": [\n"
     "      [\"ISIS\", \"released\", \"more than 200 Yazidis\"],\n"
     "      [\"Yazidis\", \"are\", \"minority group\"],\n"
     "      [\"ISIS\", \"released\", \"children and elderly Yazidis\"],\n"
     "      [\"Peshmerga commander\", \"said\", \"freed Yazidis are released\"]\n"
     "  ],\n"
     "  \"knowledge_graph2\": [\n"
     "      [\"ISIS\", \"released\", \"more than 200 Yazidis\"],\n"
     "      [\"Yazidis\", \"are\", \"minority group\"],\n"
     "      [\"Yazidis\", \"killed and displaced by\", \"ISIS\"],\n"
     "      [\"ISIS\", \"released\", \"children and elderly Yazidis\"],\n"
     "      [\"Peshmerga commander\", \"said\", \"freed Yazidis are released\"],\n"
     "      [\"Peshmerga\", \"received\", \"freed Yazidis\"],\n"
     "      [\"Peshmerga\", \"sent freed Yazidis to\", \"Irbil\"],\n"
     "      [\"Arab tribal leaders\", \"helped coordinate\", \"release of Yazidis\"]\n"
     "  ]\n"
     "}"
    },
    {"role": "user",
     "content":
     "Now extract knowledge graphs for the following 3 text(s). Return a JSON object with exactly 3 key(s): knowledge_graph1, knowledge_graph2, knowledge_graph3. Each value is a list of [subject, relation, object] triples.\n\n"
     "TEXT1:\n{paragraph_1}\n\n"
     "TEXT2:\n{paragraph_2}\n\n"
     "TEXT3:\n{paragraph_3}"
    }
]
\end{lstlisting}

## Description

This prompt is used in the KEA-based knowledge graph extraction method to generate knowledge graphs from multiple text inputs. The system includes:

- **System Role**: Defines the expert task, extraction steps, and provides two in-context examples
- **User Role**: Dynamically constructs the extraction request with the actual input paragraphs

The prompt dynamically adapts to the number of input texts (2, 3, or more), automatically generating the required number of knowledge graphs and updating the instruction message accordingly.
