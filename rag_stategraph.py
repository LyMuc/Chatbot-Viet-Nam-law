import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.prompts import ChatPromptTemplate
from IPython.display import Image, display
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

#Initiate model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(
        temperature=0, # Giảm tính ngẫu nhiên để bám sát context
        model_name="llama3-8b-8192",
        groq_api_key=""
    )

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Luat-Hon-nhan-va-gia-dinh-2014-238640.aspx",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("content1")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

vector_store = InMemoryVectorStore(embeddings)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

#Define prompt template
prompt_template = """
Bạn là một trợ lý AI chuyên tư vấn pháp luậtluật. Chỉ sử dụng thông tin được cung cấp trong phần Ngữ cảnh sau đây để trả lời câu hỏi.
Nếu câu trả lời không có trong Ngữ cảnh, hãy nói rằng bạn không tìm thấy thông tin trong tài liệu được cung cấp.
Ngữ cảnh:
{context}
Câu hỏi: {question}
Trả lời: Khi đưa ra câu trả lời, hãy trích dẫn rõ ràng điều khoản nào trong phần Ngữ cảnh là cơ sở để bạn đưa ra câu trả lời.
"""

# Define prompt for question-answering
prompt = ChatPromptTemplate.from_template(prompt_template)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile() 
