_sqlite_prompt = """You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Must pay attention to whether the columns of the called table belong to this table or not. visual_content, frame_time and audio_content belong to temporaldb. category and identification belong to instancedb.
Must check that the columns are called correctly before answering.
Must choose the temporaldb answer first, and try to call instancedb as little as possible.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

"""

PROMPT_SUFFIX = """Only use the following tables:
{table_info}

Question: {input}"""


COUNTING_EXAMPLE_PROMPT = """
Some examples of SQL queries that corrsespond to questions are:
Question: How many cats are there?
SQLQuery: SELECT COUNT(*) FROM instancedb WHERE category = 'cat';
SQLResult: "[(0,)]"
Answer: There are 0 cat.
"""

TEMPORAL_EXAMPLE_PROMPT = """
Some examples of SQL queries that corrsespond to questions are:
Question: What did the boy do after he sang a song?
SQLQuery: SELECT visual_content FROM temporaldb WHERE frame_time > (SELECT MAX(frame_time) FROM temporaldb WHERE visual_content LIKE '%Sing%') LIMIT 5;
SQLResult:  "[(" A boy ate a cake.",),(" A boy ate a cake.",),(" A boy ate a cake.",),(" A boy ate a cake.",),(" A boy ate a cake.",)]"
Answer: The boy ate a cake after he sang a song..

Question: What the guy in the video was doing when someone was dancing?
SQLQuery:SELECT visual_content FROM temporaldb WHERE visual_content LIKE '%Dance%' LIMIT 5;
SQLResult: [('a couple of people that are walking in the snow',), ('a couple of people that are standing in the snow',), ('a couple of people that are standing in the snow',), ('a couple of people that are standing in the snow',)]
Answer:The guy in the video was standing in the snow when someone was dancing.

Question: What the people in the video were doing before someone ran?
SQLQuery: SELECT visual_content FROM temporaldb WHERE frame_time > (SELECT MIN(frame_time) FROM temporaldb WHERE visual_content LIKE '%Run%') LIMIT 5;
SQLResult:  "[('There's a man and a woman walking hand in hand.',), ('There's a man and a woman walking hand in hand.',), ('There's a man and a woman walking hand in hand.',), ('There's a man and a woman walking hand in hand.',), ('There's a man and a woman walking hand in hand.',)]"
Answer: There was a man and woman walking hand in hand before someone ran.
"""

REASONFINDER_ADDITION_PROMPT = """
In addition, you are an expert at answering why. 
Before you get SQLResult to generate an Answer, 
you need to reason and analyse to find the motivation or cause behind 
an event, phenomenon or behaviour. 
Multiple factors, causality and correlation may need to be considered, 
as well as your common sense and experience.
"""

HOWSEEKER_ADDITION_PROMPT = """
In addition, you are an expert at answering how. 
Before you get SQLResult to generate an Answer, 
you need to reason and analyse to find specific action steps, methods to solve a problem.
You must give practical advice that is highly feasible, contains sequential steps, or detailed explanations.
"""

DESCRIP_ADDITION_PROMPT = """
In addition, you are an expert in description.
You must search for it directly in the corresponding attribute without combining complex statements,not including any WHERE statement. 
Question: Where could this be happening?
SQLQuery: SELECT visual_content FROM temporaldb LIMIT 5;
SQLResult:  "[('A bunch of people sitting on a bench in the living room.',), ('A bunch of people sitting on a bench in the living room.',), ('A bunch of people sitting on a bench in the living room.',), ('A bunch of people sitting on a bench in the living room.',), ('A bunch of people sitting on a bench in the living room.',)]"
Answer: This is supposed to happen in the living room.

Question: How many people are on the swing in the video?
SQLQuery: SELECT visual_content FROM temporaldb LIMIT 5;
SQLResult:  "[('Two people are on a swing in the video.',), ('Two people are on a swing in the video.',), ('Two people are on a swing in the video.',), ('Two people are on a swing in the video.',), ('Two people are on a swing in the video.',)]"
Answer: There's two guys on a swing.
"""



DESCRIP_EXAMPLE_PROMPT = """
Some examples of SQL queries that corrsespond to questions are:
Question: What did the person said?
SQLQuery: SELECT audio_content FROM temporaldb LIMIT 3;
SQLResult:  "[(" Pixar went on to create the world's first computer animated feature film, Toy Story, and",),(" Pixar went on to create the world's first computer animated feature film, Toy Story, and",),(" Pixar went on to create the world's first computer animated feature film, Toy Story, and",)]"
Answer: The boy said "Pixar went on to create the world's first computer animated feature film.".

Question: What is in the video?
SQLQuery:SELECT visual_content FROM temporaldb LIMIT 5;
SQLResult: [('a couple of people that are walking in the snow',), ('a couple of people that are standing in the snow',), ('a couple of people that are standing in the snow',), ('a couple of people that are standing in the snow',)]
Answer:The guy in the video was standing in the snow.
"""