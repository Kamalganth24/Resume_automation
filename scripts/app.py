import streamlit as st
import os
import subprocess
import sys
from dotenv import load_dotenv

# LangChain Core and Community
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.tools import Tool

from langchain.agents import create_agent

from langchain_classic.agents import AgentExecutor
from langchain_classic import hub


load_dotenv()
neo4j_user = os.getenv("neo4j_user")
neo4j_password = os.getenv("neo4j_password")
google_api_neo4j = os.getenv("google_api_neo4j")

@st.cache_resource
def start_monitor():
    subprocess.Popen([sys.executable, "monitor.py"])

start_monitor()

# PAGE SETTINGS
st.set_page_config(
    page_title="Candidate Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Candidate Analyzer Bot")

embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=google_api_neo4j
    )

# ---------------------------
# CONNECT TO NEO4J
# ---------------------------

@st.cache_resource
def connect_graph():

    graph = Neo4jGraph(
        url="bolt://localhost:7688",
        username=neo4j_user,
        password=neo4j_password,
        database="neo4j"
    )

    graph.refresh_schema()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=google_api_neo4j
    )


    cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template="""You are a Neo4j Cypher expert specializing in HR and Resume data.

### Schema:
{schema}

### Relationships:
(:Person)-[:HAS_TECH_SKILL]->(:Skill)
(:Person)-[:HAS_SOFT_SKILL]->(:SoftSkill)
(:Person)-[:WORKED_AT]->(:Company)
(:Person)-[:HELD_ROLE]->(:Role)
(:Person)-[:STUDIED_AT]->(:Education)

### Extraction & Query Rules:
1. **Semantic Search (Skills/Roles):** If a question asks for a skill or role that might have synonyms (e.g., "coding" for "Software Engineer" or "visuals" for "UI Designer"), use the Vector Index.
   - Skill Index: `skill_embeddings`
   - Role Index: `role_embeddings`
   - Syntax: `CALL db.index.vector.queryNodes('index_name', 5, $embedding) YIELD node AS n, score`

2. **Exact Matching (Names/Companies):** For specific names of people or companies, do NOT use vector search. Use `toLower` and `CONTAINS`.
   - Example: `WHERE toLower(p.name) CONTAINS toLower("Kamalganth")`

3. **Combined Queries:** When searching for a Person with a specific Skill via semantic search:
   - First, find the Skill node via `queryNodes`.
   - Then, `MATCH` the Person connected to that node.

4. **Formatting:** Always return clear property names (e.g., `p.name`, `s.name`).

### Examples:

Question: Who knows about web development?
Cypher: 
CALL db.index.vector.queryNodes('skill_embeddings', 5, $embedding) YIELD node AS s, score
MATCH (p:Person)-[:HAS_TECH_SKILL]->(s)
RETURN p.name AS Candidate, s.name AS Skill, score

Question: Find candidates who have worked as a Lead or similar.
Cypher:
CALL db.index.vector.queryNodes('role_embeddings', 5, $embedding) YIELD node AS r, score
MATCH (p:Person)-[:HELD_ROLE]->(r)
RETURN p.name AS Candidate, r.name AS Role, score

Question: What companies did Kamalganth work for?
Cypher:
MATCH (p:Person)-[:WORKED_AT]->(c:Company)
WHERE toLower(p.name) CONTAINS "kamalganth"
RETURN c.name AS Company

Question: {question}
Cypher:"""
)

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        verbose=True,
        allow_dangerous_requests=True,
        return_direct=False,
        include_embeddings=True, 
        embedding_provider=embeddings 
        )

    return chain




chain = connect_graph()

# 1. Update the Cypher Search Tool to handle embeddings
def cypher_search(query):
    print("Cypher query is exe")
    try:

        query_vector = embeddings.embed_query(query)
        result = chain.invoke({
            "query": query,
            "cypher_params": {"embedding": query_vector} # MUST HAVE THIS
        })
        print("RAW RESULT:", result)

        # Extract safely
        data = result.get("result") if isinstance(result, dict) else result

        # Convert EVERYTHING to clean string
        if isinstance(data, list):
            return "\n".join([str(row) for row in data])
        elif isinstance(data, dict):
            return str(data)
        elif data is None:
            return "No matching candidates found."
        else:
            return str(data)

    except Exception as e:
        return f"Error: {str(e)}"


# 2. Update Semantic Search (Use the existing connection settings)
def semantic_search(query):
    print("Semantic SearchQuery")
    try:
        vector_db = Neo4jVector.from_existing_index(
            embedding=embeddings,
            url="bolt://localhost:7688", 
            username=neo4j_user,
            password=neo4j_password,
            database = "neo4j",
            index_name="skill_embeddings",
            text_node_property="name",
        )
        docs = vector_db.similarity_search(query, k=3)
        if not docs:
            return "No similar skills found."
        result_str = "Found similar concepts: " + ", ".join([str(d.page_content) for d in docs])
        print(result_str)
        return result_str
    except Exception as e:
        return f"Error in Semantic Search: {str(e)}"

# 3. Define Tools 
tools = [
    Tool(
        name="Graph_Search",
        func=cypher_search,
        description="ALWAYS try this first. Use for relationships: 'Who worked at X?', 'Skills of Person Y', or finding people with specific roles/skills."
    ),
    Tool(
        name="Semantic_Search",
        func=semantic_search,
        description="Use ONLY if Graph_Search returns no results or if the user uses very vague terms like 'tech stack' or 'experience' without naming a specific skill."
    )
]


# Initialize the agent
agent = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt= """You are an expert HR and Recruitment assistant.
- Use 'Graph_Search' for structured data (who worked where, specific candidate skills).
- Use 'Semantic_Search' for finding roles or skills by meaning (e.g., 'coding for web' or 'lead roles').
- Always provide a concise, professional summary of the results."""
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True 
)



# ---------------------------
# CHAT HISTORY
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# USER INPUT
# ---------------------------

question = st.chat_input("Ask about candidates, skills, experience...")


if question:
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Analyzing intent..."):
        try:
            # The agent decides which tool to use!
            print("The response is running")
            response = agent_executor.invoke({
                "messages": [{"role": "user", "content": question}]
                })
            print(response)
            answer = response["output"]

        except Exception as e:
            answer = f"Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})