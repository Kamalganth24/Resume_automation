Overall view:
-Resume files are automatically detected from the input folder and processed by the pipeline.
-The system extracts structured information such as candidate name, skills, companies, roles, and education using LangChain with the Google Gemini API.
-This structured data is inserted into a Neo4j graph database where candidates and their relationships are stored.
-A Streamlit application provides a chat-based interface for interacting with the candidate database.
User questions are converted into Cypher queries using LangChain and executed on Neo4j to retrieve relevant candidate information.
___________________________________________________________________________________________________________
-----------------------------------------------------------------------------------------------------------

Step 1:

Open the resume automation in Vscode or any code editor
check with the folder structure.

resume_automation
│   .env
│   requirements.txt
│   resume_list.json
│   Readme.md
│
├───Incoming_folder
|       |
├───local_resume(sample for testing, not a mandatory,Resume files can be included from anywhere)
│       Account-Manager-Example-5-PDF.pdf
│       kamal_test_resume.pdf
│       My_resume (1).pdf
│       resume 5.pdf
│       test008 copy.pdf
│       test008.pdf
│       test03.pdf
│       Word-Resume-Format-with-Photo.docx
│
├───processed_folder
|       |
├───scripts
│   │   app.py
│   │   file_cleaning.py
│   │   llm_model.py
│   │   load_data_neo4j.py
│   │   main.py
│   │   monitor.py
│   │   resume_processing.py
│   │
│   └───__pycache__
│
└───state_file
        processed_files.json

------------------------------
Step 2 : (Enviromnent creation and configuration)

open Terminal:
Example : 
C:\Users\user_name\Downloads\resume_automation\resume_automation> 

Create a new Environment :
enter the following command in terminal :-
python -m venv env_name

After Entering the command you should notice the creation of new folder env_name 

Then Activate the Environment by entering following command in the terminal :-

env_name\Scripts\activate

Then you should see the environment changed from 

PS C:\Users\Name\Downloads\resume_automation\resume_automation>
to
(env_name) PS C:\Users\Name\Downloads\resume_automation\resume_automation> 

-------------------------
Step 3: (installation of necessasary libraries)

Enter the following command in terminal :

pip install  -r requirements.txt

it will take some time for installation.

after go to scripts in terminal

cd scripts

(env_name) PS C:\Users\Name\Downloads\resume_automation\resume_automation\scripts> 

And run the following command in terminal after setting up neo4j.

Command: streamlit run app.py

________________________________________________________
Settingup Local Neo4j:

-Option1: Docker Image
-Option2: Neo4j Desktop

Option1: 
        Ensure Docker is installed and running on your system.

        Run the following command in the terminal to start a Neo4j container with the APOC plugin enabled:
        
        docker pull neo4j:5.18

        docker run -d --name neo4j-apoc -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -e "NEO4J_PLUGINS=[\"apoc\"]" neo4j:5.18

        Users should replace neo4j/password with their preferred username and password.

Option2:
        Install Neo4j Desktop from the official website: https://neo4j.com/download/

        Open Neo4j Desktop and create a new project.

        Inside the project, create a new local database.

        Start the database instance.

        Set a username and password for the database.


After setting up (Option1/2):

        Open Neo4j browser at : http://localhost:7474

        Login to Neo4j using the credentials.

        Ensure the Neo4j credentials in the .env file match the credentials used when starting the Neo4j container/desktop.

        The application will now be able to connect to the local Neo4j database.

 ____________________________________________________________________________________________________________

Chat session can be accessed through streamlit's locallyhosted browser tab once you dropped the resume into Incoming_folder.

Example Input/Output:

Question input : What are Jacky Wilson skills
Expected Output: Jacky Wilson have skills of Graphic design, Microsoft Word.

Returns No data found for irrelevant questions or user data not in database.

NOTE!!- Update your API key in .env file, Change your API key if quota exhausted.

    "role_search","role_search","role_search","role_search",
    "contact_info","contact_info","contact_info","contact_info",
    "non_tech_search","non_tech_search","non_tech_search","non_tech_search",
    "semantic_search","semantic_search","semantic_search","semantic_search"
]
