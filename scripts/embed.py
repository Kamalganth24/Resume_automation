import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI



load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0
    )

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def chunking(resume_text_list, max_words=50):
    if not resume_text_list:  # check if list is empty
        print("No resumes found to chunk.")
        return []
    all_chunks = []
    for resume in resume_text_list:
        resume_text = resume["text"]
        print("Embedding resume:", resume["file"])
        sentences = sent_tokenize(resume_text)
        current, count = [], 0
        for s in sentences:
            words = s.split()
            if count + len(words) > max_words:
                all_chunks.append(" ".join(current))
                current, count = [], 0
            current.append(s)
            count += len(words)
        if current:
            all_chunks.append(" ".join(current))
    return all_chunks


def build_index(resume_text_list):
    print("chunking")
    chunks = chunking(resume_text_list, max_words=50)
    if not chunks:  # if chunking returned empty
        print("No chunks created. Returning empty index.")
        dimension = 768  # default dimension for all-mpnet-base-v2
        index = faiss.IndexFlatIP(dimension)
        return index, []
    
    print("chunk done")
    embeddings = model.encode(chunks)
    print(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    return index, chunks

def query_with_llm(query, index, chunks, k=3):
    # Retrieve top-k chunks from FAISS
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    indices = index.search(query_embedding.astype("float32"), k=k)

    retrieved_chunks = [chunks[idx] for idx in indices[0] if idx != -1]

    # Build prompt for Gemini
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    You are a resume assistant.
    User query: {query}
    Resume data:
    {context}

    Please answer clearly. 
    - If the query asks for candidates, list names. 
    - If it asks for contact info, return email/phone. 
    - If it asks for skills or roles, filter accordingly. 
    - Otherwise, summarize relevant resumes.
    """

    response = llm.invoke(prompt)
    return response.content

# -------------------------------
# Test block
# -------------------------------
if __name__ == "__main__":
    # Sample resume text list
    sample_resumes = [
  {
    "file": "My_resume (1).pdf",
    "text": "KamalganthS +919445171303|kamalganths2004@gmail.com |Github |LinkedInTechnicalSummaryAI & DataScienceengineeringstudentwithhands-onexperienceinGenerativeAI, multi-agentsystems, andMLpipelines.Skilledinbuildingandfine-tuningAImodelsforreal-worldapplicationsincludingNLPandagenticworkflows.EducationSriEshwarCollegeofEngineeringCoimbatore, IndiaB.TechinArtificialIntelligence & DataScience;CGPA: 7.9/10.0 (7thSem)Aug2022 May2026AdhiyamanMatricHrSecSchoolTamilNadu, IndiaHSC 89.7%|SSLC 94.8%2020 2022|2019 2020ExperienceGenerativeAIIntern2024GlobalKnowledgeRemote Builtandfine-tunedAImodelsfortextandimagegeneration; deepenedexpertiseinNLP, transformerarchitectures, andreal-worldAIintegrationusingPython, TensorFlow, HuggingFaceTransformers, andOpenAIAPIs.AIMarketResearchInternSep.2025 Nov.2025M/s.FreeThinkers (Brand Breakout ) Bangalore/Remote ConductedAImarketresearchandanalysis; gainedexposuretoreal-worldAImarketdynamicsandbusinessintelligenceworkflowsoverafocused2-monthengagement.OpenSourceContributor2025Hive aden-hive/hive (GitHub) Remote Reported & documentedIssue #3349; integratedOpenAIWhisperreducingmanualtranscriptiontimeby80%, achievingWER8-12%oncleanEnglishspeech, andresolvedFP16CPUfallbackwithdocumentedfix.ProjectsAdaptiveCEOAvatarSystem (Gen-AIMVP)|Python, Streamlit, LLMs, HeyGenAPI, orca:mini2026 Builtanend-to-endGen-AIsystemthatconvertsstructuredbusinesscontextintorealisticexecutiveavatarvideosusingLLMsandHeyGenstext-to-videoAPI.Reducedscriptgenerationlatencyto2sandachievedappx.93%semanticalignmentaccuracywhileloweringtokencostbyappx.20%throughpromptoptimization.LLMResponseEvaluationPipeline|Python, TF-IDF, SentenceEmbeddings, Gemini, DeepEval, RAG2025 Builtamodular, end-to-endLLMevaluationpipelinecombiningTF-IDF, embeddingsimilarity, GeminiLLMjudge, andDeepEvaltoassessrelevance, completeness, andhallucinationacrossRAGoutputs.ImprovedRAGrelevanceby18%andreducedhallucinationratefrom27%to11%whilebenchmarkingevaluationlatency (2-6s) andcostperresponse.AIAgent StreamlitChat|Python, Streamlit, LangChain, LangGraph, Gemini, Docker2025 BuiltalightweightStreamlitchatapplicationpoweredbyGoogleGemini (viaLangChain/LangGraph), exposingamodulartoolsetincludingacalculator, unitconverter, symbolicmathsolver, andlivenewsfetcher.Achievedappx.96%tool-selectionaccuracywith700mstoolinvocationlatencyandsupported20concurrentsessionsincontainerizeddeployment.MiningOperationsManagementSystem|Python, Streamlit, Scikit-learn, Pandas, NumPy, Matplotlib, SQLite2025 BuiltanAI-enabledsystemforpredictivemaintenance, productionforecasting, workerproductivityinsights, andautomatedenvironmentalsafetyalertswithsmartchatbotintegration.Delivered87%predictivemaintenanceaccuracy, reduceddowntimepredictionerrorby21%, andimprovedforecastingMAEby18%afterfeatureengineering.AchievementsCompetitiveProgramming: LeetCode40+|CodeChef50+|HackerRank Python(Gold), Java(Gold), SQL (Silver) Hackathons: IgniteInnovations2.0 1stPlace/Rs.2000Prize|RevaHackathon Cleared2rounds/50teams|SECEProjectExpo 3rdPrizeCertificationsProgramming & CS: CProgramming (Udemy)|PythonbyGoogle (Coursera)|DataStructuresinC/C++ (Udemy) MachineLearning: SupervisedML: RegressionandClassification (Coursera) Platform & Tools: Micro-Certification WelcometoServiceNowXanadu (ServiceNow)|Skills: KnowledgeManagement, ServiceNowAIPlatform, VisualTaskBoards, ServiceCatalog, PlatformAnalyticsLeadership & ResponsibilitiesStudentCoordinator AIContentCreator (TechnicalEvent): Co-organizedatwo-roundAIcontentcreationeventevaluatedfororiginality, coherence, andcross-roundthemeconsistency; managedparticipantengagementandeventlogistics.WorkshopSpeaker DataScience & BigData (SasurieCollegeofEngineering): Conductedatechnicalworkshopon DataScience & BigData: FoundationsofModernTechnologicalAdvancements, coveringcoreconcepts, real-worldapplications, andemergingindustrytrends; engagedparticipantsthroughinteractivediscussionsandpracticalinsights.TechnicalSkillsLanguages: Python, JavaDatabase: SQL, SQLite, MongoDB, VectorDBCoreConcepts: OOPs, OS, DBMSAI/MLTechnologies: MachineLearning, GenerativeAI, NLP, AgenticAI, DeepLearning, RAG, PromptEngineeringFrameworks & Libraries: LangChain, LangGraph, FAISS, FastAPI, Streamlit, HuggingFaceTransformers, Scikit-learn, TensorFlow, DeepEval.Tools & APIs: OpenAIAPIs, HeyGenAPI, GeminiAPI, Neo4j, Git, GitHub, Docker"
  },
  {
    "file": "test008 copy.pdf",
    "text": "JasonMillerAmazonAssociateProfileExperiencedAmazonAssociatewithfiveyears tenureinashippingyard setting, maintaininganaveragepicking/packingspeedof98%.Holdsa zeroerror% scoreinadheringtopackingspecsand97% error-freeratio onpackingrecords.CompletedacertificateinWarehouseSanitationand hasavalidcommercialdriverslicense.EmploymentHistoryAmazonWarehouseAssociateatAmazon, MiamiGardensJanuary2021 July2022Performedallwarehouselaborerdutiessuchaspacking, picking, counting, recordkeeping, andmaintainingacleanarea.Consistentlymaintainedpicking/packingspeedsinthe98th percentile.Pickedallorderswith100% accuracydespitehighspeeds.Maintainedacleanworkarea, meeting97.5% oftheinspection requirements.LaboratoryInventoryAssistant at DunreaLaboratories, OrlandoJanuary2019 December2020Full-timelabassistantinasmall, regionallaboratorytaskedwith participatinginKaizenEvents, Gembawalks, and5Storemovebarriers andimproveproductivity.Filledthewarehousehelperjobdescription, whichinvolved picking, packing, shipping, inventorymanagement, andcleaning equipment.Saved12% onUPSordersbystayingontopofspecialdeals.Cutdownstoragewasteby23% byswitchingtoaKanbansystem.EducationAssociatesDegreeinLogisticsandSupplyChainFundamentals, AtlantaTechnicalCollege, AtlantaJanuary2021 July2022 Majors: WarehousingOperations, LogisticsandDistribution Practices Minors: InventorySystems, SupplyChainPrinciplesCoursesOnlineGraduateCertificateinWarehousing & SupplyChain Management, SouthernNewHampshireUniversity (SNHU), NH.July2022 July2022Details1515PacificAveLosAngeles, CA90291UnitedStates3868683442email@email.comPlaceofbirthSanAntonioDrivinglicenseFullLinksLinkedInPinterestResumeTemplatesBuildthistemplateSkillsCleaningEquipmentCleaningEquipmentMathematicsCleaningEquipmentDeepSanitationPracticesHobbiesActionCricket, Rugby, AthleticsLanguagesEnglishSpanishWarehousing, Operations, andDisposalCourse, GraduateSchool USA, WashingtonDC.January2021 May2021Achievements DecreasetheerrorsrateandQCseveralmoreordersthatweship perday.Awarded \"Employeeofthemonth\" duetoconsistentattendance, punctuality, andperformance.ManagedworkflowofassociatesfortheFC."
  }
    ]
    index, chunks = build_index(sample_resumes)

    query = "Who are all best fit for Developer Role"
    answer = query_with_llm(query, index, chunks)
    print("\nLLM Answer:\n", answer)

    