import os
# from dotenv import load_dotenv
# from groq import Groq


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os
from langchain_groq import ChatGroq
from langchain import hub
import streamlit as st

# streamlit
st.set_page_config(
    page_title = "PDF Q&A"
    ,page_icon = "🖹"
    ,layout = "centered"
)

# initialize chat history if not already there
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# page title
st.title("🖹 PDF Q&A")

# chat history display
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# inputs
report = st.sidebar.text_input("Please enter a link to your PDF document:")
user_prompt = st.chat_input("Ask a question about the PDF")

# load the rag vector db or create it

# load_dotenv()

wd = os.getcwd()

# Load, chunk and index the contents of the 10-q.
# report = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/faab4555-c69b-438a-aaf7-e09305f87ca3.pdf"
pattern = r'/([^/]+)\.pdf$'

if report: # if the report has been loaded by the user
    st.sidebar.write(f"The url provided was: {report}")
    st.sidebar.write("Thanks! Checking if a Chroma vectordb already exists - if not, one will be created for this document")
    match = re.search(pattern, report)
    report_name = match.group(1)

    # initialize the loader using the report URL
    loader = PyPDFLoader(report)

    # load the file
    docs = loader.load()

    # initialize the text splitter specifying 1k size chunks with 250 size overlap. Add indexing to metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250, add_start_index=True)

    # split the document
    docs_split = text_splitter.split_documents(docs)

    # Create the vector store only if it doesn't already exist
    if os.path.exists('./' + report_name):
        vectorstore = Chroma(persist_directory='./' + report_name, embedding_function=OpenAIEmbeddings())
        # print("Found existing db: {}".format(report_name))
        st.sidebar.write("Found existing db: {}".format(report_name))
    else:
        vectorstore = Chroma.from_documents(documents=docs_split
                                        , embedding=OpenAIEmbeddings()
                                        ,persist_directory = './' + report_name
                                        ) 
        # print("No existing db, creating new db: {}".format(report_name))
        st.sidebar.write("No existing db, creating new db: {}".format(report_name))

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult":0.5} )

    llm = ChatGroq(model="llama-3.1-8b-instant")

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    rag_chain = (
    RunnablePassthrough.assign(
        context=lambda params: format_docs(params["context"]),
        answer=lambda params: params["answer"],
    )
    | prompt
    | llm
    | StrOutputParser()
)

# Get the user input
if user_prompt:
    # print(f"user_prompt type is: {type(user_prompt)}")
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role":"user","content":user_prompt})

    # response = rag_chain.invoke(user_prompt)
    response = rag_chain.invoke({"question": user_prompt, "context": retriever.get_relevant_documents(user_prompt)})

    # st.session_state.chat_history.append({"role":"assistant","content":response})
    st.session_state.chat_history.append({"role":"assistant","content":response["answer"] + '\n Context: \n' + response["context"]})

    # llm response
    with st.chat_message("assistant"):
        st.markdown(response["answer"] + '\n Context: \n' + response["context"])


# for chunk in rag_chain.stream("Were there any audit issues identified or financial reporting inaccuracies?"):
#     print(chunk, end="", flush=True)