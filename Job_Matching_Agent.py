from langchain_core.documents import Document

import pdfplumber
import streamlit as st
import pdfplumber as pdf
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import vector_stores
import re
import json

from sqlalchemy.testing.suite.test_reflection import metadata

from main import fetch_jobs

st.markdown("""
<style>
.job-card {
    padding: 20px;
    border-radius: 10px;
    background-color: #f5f7fa;
    margin-bottom: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("💼 AI Job Matching Agent")
st.markdown("Find the most relevant jobs using AI-powered matching.")
st.divider()

st.sidebar.title("Filters")
experience = st.sidebar.selectbox(
    "Experience Level",
    ["Any", "0-2 Years", "3-5 Years", "5+ Years"]
)

st.set_page_config(
    page_title="AI Job Matcher",
    page_icon="💼",
    layout="wide"
)

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("🔍 Enter your skills or role")

with col2:
    search_button = st.button("Search Jobs")

file = st.file_uploader(
    "Uploaded File",
    help="Uploaded File",
    type="pdf"
)

if file is not None:
    with st.spinner("Processing file..."):
        with pdfplumber.open(file) as pdf:
            text=""
            for temp in pdf.pages:
                text+=temp.extract_text()

    with st.spinner("Summarizing Resume and extracting details ..."):
        llm = Ollama(
            model="llama3",
            temperature=0.0
        )

        prompt = f"""
        You are a JSON generator.
    
        CRITICAL INSTRUCTIONS (must be followed exactly):
        - Output ONLY a valid JSON object
        - Do NOT include any explanations
        - Do NOT include markdown
        - Do NOT include backticks
        - Do NOT include any text before or after the JSON
        - The response MUST start with '{{' and MUST end with '}}'
        - If any field is missing, return an empty value for it
    
        Return the JSON strictly in this schema:
        {{
          "skills": [],
          "years_of_experience": "",
          "preferred_roles": [],
          "location": "",
          "domain": ""
        }}
    
        Resume text:
        {text}
        """

        response = llm.invoke(prompt)

   # text_splitter = RecursiveCharacterTextSplitter(
   #     separators=["\n\n", "\n", ". ", ", ", ""],
   #     chunk_size=500,
   #     chunk_overlap=100
   # )
   # chunks = text_splitter.split_text(response)
    with st.spinner("Converting to embeddings"):
        embedding_model = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1",
            model_kwargs={"trust_remote_code": True}
        )

        resume_embeddings = embedding_model.embed_query(text)

    with st.spinner("Fetching the relevant jobs for you !!"):
        import json
        #
        # print("TYPE:", type(response))
        # print("RAW RESPONSE START")
        # print(repr(response))
        # print("RAW RESPONSE END")
        response = json.loads(response)
        keyword = response['skills'][0] + " jobs in " + response['location']
        print(keyword + "\n\n")
        job_list = fetch_jobs(keyword)
        # print(job_list)
        # print("\n\n")

    with st.spinner("Collecting metadata ... "):
        job_texts = [
            job["job_title"] + " " + job["job_description"]
            for job in job_list["data"]
        ]
        print(job_texts)
        print("\n\n")

        documents = []

        for job in job_list["data"]:
            searchable_text = job["job_title"] + " " + job["job_description"]

            documents.append(
                Document(
                    page_content=searchable_text,  # used for embedding
                    metadata={
                        "job_id": job.get("job_id"),
                        "job_title": job.get("job_title"),
                        "job_description": job.get("job_description"),
                        "job_apply_link": job.get("job_apply_link"),
                        "job_is_remote": job.get("job_is_remote"),
                        "job_city": job.get("job_city"),
                        "job_state": job.get("job_state"),
                        "raw_json": job  # keep full structured data
                    }
                )
            )
    with st.spinner("Finding best job matches..."):
        vector = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model
        )

        jobs_matches = vector.similarity_search_by_vector(
            resume_embeddings,
            k=3
        )

    with st.spinner("Displaying matches ... "):
        for doc in jobs_matches:
            jid = doc.metadata.get("job_id")
            title = doc.metadata.get("job_title")

            st.markdown(f"""
            <div class="job-card">
            <h3>{title}</h3>
            <p><b>Job ID:</b> {jid}</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View Job Description"):
                st.write("Job Description : " + doc.metadata["job_description"])
                st.write("\n\n")
                st.write("Apply Here : " + doc.metadata["job_apply_link"])
                st.write("\n\n")
                st.write("City : " + doc.metadata["job_city"])
                st.write("\n\n")
                st.write("State : " + doc.metadata["job_state"])
                st.write("\n\n")

            st.divider()



