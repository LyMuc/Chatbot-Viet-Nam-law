import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, MessagesState, END
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.prompts import ChatPromptTemplate
from IPython.display import Image, display
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import LongContextReorder


#Initiate model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(
        temperature=0, # Giảm tính ngẫu nhiên để bám sát context
        model_name="llama3-8b-8192",
        groq_api_key=""
    )

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://thuvienphapluat.vn/van-ban/Quyen-dan-su/Luat-Hon-nhan-va-gia-dinh-2014-238640.aspx",
               "https://thuvienphapluat.vn/van-ban/Thu-tuc-To-tung/Nghi-quyet-01-2024-NQ-HDTP-huong-dan-ap-dung-giai-quyet-vu-viec-ve-hon-nhan-gia-dinh-515531.aspx",
               "https://thuvienphapluat.vn/van-ban/Hon-nhan-Gia-dinh/Nghi-dinh-126-2014-ND-CP-huong-dan-Luat-Hon-nhan-va-gia-dinh-247904.aspx",
               "https://thuvienphapluat.vn/van-ban/Hon-nhan-Gia-dinh/Thong-tu-lien-tich-01-2016-TTLT-TANDTC-VKSNDTC-BTP-huong-dan-thi-hanh-Luat-hon-nhan-gia-dinh-300182.aspx",
               "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Nghi-dinh-98-2016-ND-CP-sua-doi-10-2015-ND-CP-sinh-con-thu-tinh-trong-ong-nghiem-mang-thai-ho-315458.aspx",
               "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Nghi-dinh-10-2015-ND-CP-sinh-con-bang-ky-thuat-thu-tinh-trong-ong-nghiem-mang-thai-ho-264622.aspx",
               ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("content1")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
all_splits = text_splitter.split_documents(docs)

#vector_store = InMemoryVectorStore(embeddings)

# Index chunks
#_ = vector_store.add_documents(documents=all_splits)

vector_store = FAISS.from_documents(all_splits, embedding=embeddings)


#Build retrieve tool for generating more concise query
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Truy xuất thông tin pháp lý liên quan đến câu hỏi từ các văn bản Luật Hôn nhân và Gia đình."""
    docs = vector_store.similarity_search(query, k=4)
    reordering= LongContextReorder()
    retrieved_docs = reordering.transform_documents(docs)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """
    Bạn là một chuyên gia phân tích câu hỏi người dùng về Luật Hôn nhân và Gia đình Việt Nam.
    Nhiệm vụ của bạn là đọc câu hỏi cuối cùng của người dùng và quyết định:
    1.  Nếu câu hỏi rõ ràng cần tra cứu thông tin trong luật (ví dụ: hỏi về điều kiện kết hôn, quyền ly hôn, tài sản chung,...), hãy tạo một **cụm từ tìm kiếm (query) ngắn gọn, súc tích và thật cụ thể** chỉ chứa các từ khóa chính liên quan trực tiếp đến nội dung cần tìm trong luật. Ví dụ: "điều kiện kết hôn", "thủ tục ly hôn đơn phương", "tài sản riêng vợ chồng", "quyền yêu cầu ly hôn". Đừng thêm các từ như "luật", "Việt Nam", "cho tôi biết",... vào query. Sau đó gọi tool `retrieve` với query này.
    2.  Nếu câu hỏi mang tính chào hỏi, cảm ơn, hoặc không rõ ràng cần tra cứu luật, hãy tạo một câu trả lời trực tiếp phù hợp.
    """
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}
# Step 2: Execute the retrieval

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Generate answer"""
    #Get generated tool messages for prompt
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else: break

    tool_messages = recent_tool_messages[::-1]
    
    #format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        """Bạn là một trợ lý ảo AI tư vấn pháp luật chuyên nghiệp. Dựa vào ngữ cảnh cung cấp dưới đây, hãy đưa ra câu trả lời cho các câu hỏi của người dùng. 
           Nếu câu hỏi chứa những thông tin mơ hồ và không cụ thể như trong các thông tin được cung cấp cho bạn, hãy trả lời là không biết hoặc hỏi lại người dùng để xác định chính xác những thông tin mơ hồ. 
           Mỗi khi trả lời, điều đầu tiên phải làm là chỉ rõ bạn căn cứ vào điều luật nào của bộ luật nào rồi sau đó mới đưa ra câu trả lời. Nếu câu trả lời chỉ căn cứ vào 1 khoản hoặc 1 mục của 1 điều luật thì nêu chính xác cả điều khoản, mục đó ra; còn không thì chỉ nêu tên điều luật. 
           
           **Lưu ý quan trọng:** Câu trả lời phải trả lời tất cả các phần của câu hỏi, và chỉ dựa vào những phần ngữ cảnh có vẻ liên quan trực tiếp đến từng phần của câu hỏi. Nếu không tìm ra được các ngữ cảnh chính xác tương ứng với các phần của câu hỏi thì trả lời là không biết.
           **Lưu ý quan trọng về ngôn ngữ:** Toàn bộ câu trả lời, bao gồm cả lời mở đầu, câu chuyển tiếp, lời xin lỗi (nếu có) và nội dung chính, phải được viết hoàn toàn bằng **tiếng Việt**.
        """
        "Ngữ cảnh:" f"{docs_content}"
    )
    
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    
    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    
    return {"messages": [response]}

#Build Graph
graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition, 
    {
        "tools": "tools",
        END: END,
    }
)

graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "no1"}}



