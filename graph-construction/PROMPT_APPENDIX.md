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
     "4. Knowledge Graph refinement: Where the same entity or relation appears across multiple graphs, use the same label consistently. Do not merge distinct facts — preserve each triplet independently.\n\n"
     "Format your response as a JSON object. Do not include any text outside the JSON.\n"
     "Each knowledge graph is a list of triples: [[\"subject\", \"relation\", \"object\"], ...].\n\n"
     "EXAMPLE 1:\n"
     "TEXT1 (reference answer):\n"
     "A&E Networks will simulcast the original \"Roots\" in 2016. The original \"Roots\" premiered in 1977 and ran for four seasons. The miniseries followed Kunta Kinte, a free black man in Virginia, as he was sold into slavery.\n\n"
     "TEXT2 (model-generated response):\n"
     "(CNN)One of the biggest TV events of all time is being reimagined for new audiences. \"Roots,\" the epic miniseries about an African-American slave and his descendants, had a staggering audience of over 100 million viewers back in 1977\n\n"
     "TEXT3 (supporting context):\n"
     "Roots is a 1977 American television miniseries based on Alex Haley's novel. It depicts the life of Kunta Kinte, an African who was captured and sold into slavery in America. The series was broadcast on ABC and became one of the most watched programs in US television history.\n\n"
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
     "  ],\n"
     "  \"knowledge_graph3\": [\n"
     "      [\"Roots\", \"is a\", \"American television miniseries\"],\n"
     "      [\"Roots\", \"based on\", \"Alex Haley's novel\"],\n"
     "      [\"Roots\", \"depicts life of\", \"Kunta Kinte\"],\n"
     "      [\"Kunta Kinte\", \"was captured and sold into\", \"slavery\"],\n"
     "      [\"Roots\", \"was broadcast on\", \"ABC\"],\n"
     "      [\"Roots\", \"premiered in\", \"1977\"],\n"
     "      [\"Roots\", \"one of the most watched\", \"US television programs\"]\n"
     "  ]\n"
     "}\n\n"
     "EXAMPLE 2:\n"
     "TEXT1 (reference answer):\n"
     "ISIS released more than 200 Yazidis, a minority group, a group says. The Islamist terror group has been killed in recent summer. ISIS released scores of other Yazidis, mainly children and the elderly. The Peshmerga commander says the freed Yazidis are released.\n\n"
     "TEXT2 (model-generated response):\n"
     "(CNN) ISIS on Wednesday released more than 200 Yazidis, a minority group whose members were killed, captured and displaced when the Islamist terror organization overtook their towns in northern Iraq last summer, officials said. Most of those released were women and children; the rest were ill or elderly, said Rassol Omar, a commander in the Peshmerga force that defends northern Iraq's semi-autonomous Kurdish region. Omar didn't say what led to the release, other than asserting that Arab tribal leaders helped to coordinate it. The freed Yazidis were received by Peshmerga, who sent them to the Kurdish regional capital, Irbil, said Nuri Osman, an official with Iraq's Kurdistan Regional Government. It wasn't immediately clear what motivated Wednesday's release, Osman said.\n\n"
     "TEXT3 (supporting context):\n"
     "The Yazidis are a Kurdish-speaking minority group native to northern Iraq. They follow a religion that combines elements of several ancient traditions. In August 2014, ISIS launched a major offensive against Yazidi communities in the Sinjar region, killing hundreds and enslaving thousands. The attack prompted international condemnation and led to US airstrikes against ISIS positions in northern Iraq.\n\n"
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
     "  ],\n"
     "  \"knowledge_graph3\": [\n"
     "      [\"Yazidis\", \"are\", \"Kurdish-speaking minority group\"],\n"
     "      [\"Yazidis\", \"native to\", \"northern Iraq\"],\n"
     "      [\"ISIS\", \"launched offensive against\", \"Yazidi communities\"],\n"
     "      [\"ISIS\", \"offensive in\", \"Sinjar region\"],\n"
     "      [\"ISIS\", \"killed hundreds of\", \"Yazidis\"],\n"
     "      [\"ISIS\", \"enslaved thousands of\", \"Yazidis\"],\n"
     "      [\"attack\", \"prompted\", \"international condemnation\"],\n"
     "      [\"attack\", \"led to\", \"US airstrikes against ISIS\"]\n"
     "  ]\n"
     "}"
    },
    {"role": "user",
     "content":
     "Now extract knowledge graphs for the following 3 text(s). Return a JSON object with exactly 3 key(s): knowledge_graph1, knowledge_graph2, knowledge_graph3. Each value is a list of [subject, relation, object] triples.\n\n"
     "TEXT1 (reference answer):\n{paragraph_1}\n\n"
     "TEXT2 (model-generated response):\n{paragraph_2}\n\n"
     "TEXT3 (supporting context):\n{paragraph_3}"
    }
]
\end{lstlisting}

## Description

This prompt is used in the KEA-based knowledge graph extraction method to generate knowledge graphs from multiple text inputs. The system includes:

- **System Role**: Defines the expert task, extraction steps, and provides two in-context examples
- **User Role**: Dynamically constructs the extraction request with the actual input paragraphs

The prompt dynamically adapts to the number of input texts (2, 3, or more), automatically generating the required number of knowledge graphs and updating the instruction message accordingly. The examples demonstrate extraction from three text types: reference answers, model-generated responses, and supporting context.
