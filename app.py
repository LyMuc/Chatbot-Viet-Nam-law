import streamlit as st

# Lưu lịch sử hội thoại trong session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập câu hỏi người dùng
user_input = st.chat_input("Hỏi tôi về luật hôn nhân và gia đình...")

if user_input:
    # Hiển thị câu hỏi người dùng lên màn hình
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Gửi vào hệ thống RAG của bạn để lấy câu trả lời
    from rag import graph  # import từ file chứa graph bạn đã build
    config = {"configurable": {"thread_id": "no1"}}
    output = None

    for step in graph.stream({"messages": [{"role": "user", "content": user_input}]},
                             stream_mode="values", config=config):
        output = step["messages"][-1].content

    # Hiển thị phản hồi AI
    with st.chat_message("assistant"):
        st.markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})
