from neo4j import GraphDatabase
from llm_model import extract_all_resumes
from sentence_transformers import SentenceTransformer



model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def run_resume_graph_pipeline(llm_response):

    

    URI = "bolt://localhost:7687"
    USERNAME = "neo4j"
    PASSWORD = "password"

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


    def create_schema(session):
    # Existing constraints
        constraints = [
        "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT softskill_name IF NOT EXISTS FOR (s:SoftSkill) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT role_name IF NOT EXISTS FOR (r:Role) REQUIRE r.name IS UNIQUE",
        "CREATE CONSTRAINT education_name IF NOT EXISTS FOR (e:Education) REQUIRE e.name IS UNIQUE"
        ]
    
    # New Vector Indexes
        vector_indexes = [
        """
        CREATE VECTOR INDEX skill_embeddings IF NOT EXISTS
        FOR (s:Skill) ON (s.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 768,
          `vector.similarity_function`: 'cosine'
        }}
        """,
        """
        CREATE VECTOR INDEX role_embeddings IF NOT EXISTS
        FOR (r:Role) ON (r.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 768,
          `vector.similarity_function`: 'cosine'
        }}
        """
        ]

        for query in constraints + vector_indexes:
            session.run(query)



    def insert_data(session, data):
        for person in data:
            name = person["name"]
        
        # Ensure the Person exists first
            session.run("MERGE (p:Person {name:$name})", name=name)

        # 1. Technical Skills with Embeddings
            for skill in person["skills"]["technical"]:
            # encode().tolist() is correct for mpnet-base-v2 (768 dims)
                embedding = model.encode(skill).tolist() 
            
                session.run("""
                MERGE (s:Skill {name:$skill})
                SET s.embedding = $embedding  // Changed from ON CREATE SET to SET
                WITH s
                MATCH (p:Person {name:$name})
                MERGE (p)-[:HAS_TECH_SKILL]->(s)
                """, name=name, skill=skill, embedding=embedding)

        # 2. Roles with Embeddings
            for role in person["roles"]:
                embedding = model.encode(role).tolist()
                session.run("""
                MERGE (r:Role {name:$role})
                SET r.embedding = $embedding  // Changed from ON CREATE SET to SET
                WITH r
                MATCH (p:Person {name:$name})
                MERGE (p)-[:HELD_ROLE]->(r)
                """, name=name, role=role, embedding=embedding)

        # 3. Soft Skills (Standard MERGE)
            for skill in person["skills"]["soft"]:
                session.run("""
                MATCH (p:Person {name:$name})
                MERGE (s:SoftSkill {name:$skill})
                MERGE (p)-[:HAS_SOFT_SKILL]->(s)
                """, name=name, skill=skill)

        # 4. Companies
            for company in person["companies"]:
                session.run("""
                MATCH (p:Person {name:$name})
                MERGE (c:Company {name:$company})
                MERGE (p)-[:WORKED_AT]->(c)
                """, name=name, company=company)

        # 5. Education
            for edu in person["education"]:
                session.run("""
                MATCH (p:Person {name:$name})
                MERGE (e:Education {name:$edu})
                MERGE (p)-[:STUDIED_AT]->(e)
                """, name=name, edu=edu)





    print("Extracting structured data using Gemini...")

    extracted_data=llm_response ###########


    with driver.session() as session:

        print("Creating schema...")
        create_schema(session)

        print("Inserting extracted data into Neo4j...")
        insert_data(session, extracted_data)

        print("Data successfully inserted!")


    # Start LangChain Query System
   