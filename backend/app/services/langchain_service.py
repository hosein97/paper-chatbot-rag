from app.config.settings import VECTOR_DB_PATH, OPENAI_API_KEY
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings



llm = ChatOpenAI(model="gpt-4o-mini")

memory = MemorySaver()


def process_question(filename: str, question: str) -> str:
    """
    Process a user's question and maintain chat history using MemorySaver.
    """
    # Load the vector database
    vector_store = Chroma(persist_directory=VECTOR_DB_PATH,
                           collection_name="file_collection",
                           embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
                           )

    graph_builder = StateGraph(MessagesState)

    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve information of the paper."""
        retrieved_docs = vector_store.similarity_search(query, k=2)
        print(retrieved_docs[0])
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        system_prompt = (
            "You are an assistant answering questions about a paper stored in a database. "
            "If the question relates to the paper's content, call the `retrieve` tool to fetch context. "
            "Otherwise, respond directly."
        )
        conversation = [SystemMessage(system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(conversation)
        return {"messages": [response]}

    def generate(state: MessagesState):
        """Generate answer."""
        recent_tool_messages = [
            message for message in reversed(state["messages"]) if message.type == "tool"
        ][::-1]
        docs_content = "\n\n".join(doc.content for doc in recent_tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        prompt = [SystemMessage(system_message_content)] + [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        response = llm.invoke(prompt)
        return {"messages": [response]}

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(ToolNode([retrieve]))
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    # Compile the graph with memory for chat history
    graph = graph_builder.compile(checkpointer=memory)

    # Use a unique thread ID for the session
    thread_id = filename  # Associate thread ID with the file ID
    config = {"configurable": {"thread_id": thread_id}}

    # Process the question
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    return result["messages"][-1].content



# def process_question(file_id: str, question: str) -> str:
#     """
#     Search within the vector database for answers to the user's question, filtered by file_id.
#     """
#     # Load the vector database
#     vector_store = Chroma(persist_directory=VECTOR_DB_PATH)



#     graph_builder = StateGraph(MessagesState)


#     @tool(response_format="content_and_artifact")
#     def retrieve(query: str):
#         """Retrieve information of the paper."""
#         retrieved_docs = vector_store.similarity_search(query, k=2)
#         serialized = "\n\n".join(
#             (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
#             for doc in retrieved_docs
#         )
#         return serialized, retrieved_docs

#     # Step 1: Generate an AIMessage that may include a tool-call to be sent.
#     def query_or_respond(state: MessagesState):
#         """Generate tool call for retrieval or respond."""
#         llm_with_tools = llm.bind_tools([retrieve])
#         # response = llm_with_tools.invoke(state["messages"])
#         # # MessagesState appends messages to state instead of overwriting
#         # return {"messages": [response]}
#         system_prompt = (
#             "You are an assistant answering questions about a paper stored in a database. "
#             "If the question relates to the paper's content, call the `retrieve` tool to fetch context. "
#             "Otherwise, respond directly."
#         )
#         conversation = [
#             SystemMessage(system_prompt)
#         ] + state["messages"]

#         response = llm_with_tools.invoke(conversation)
#         print("Generated response:", response)  # Debugging
#         return {"messages": [response]}



#     # Step 2: Execute the retrieval.
#     tools = ToolNode([retrieve])


#     # Step 3: Generate a response using the retrieved content.
#     def generate(state: MessagesState):
#         """Generate answer."""
#         # Get generated ToolMessages
#         recent_tool_messages = []
#         for message in reversed(state["messages"]):
#             if message.type == "tool":
#                 recent_tool_messages.append(message)
#             else:
#                 break
#         tool_messages = recent_tool_messages[::-1]

#         # Format into prompt
#         docs_content = "\n\n".join(doc.content for doc in tool_messages)
#         system_message_content = (
#             "You are an assistant for question-answering task. "
#             "Use the following pieces of retrieved context to answer "
#             "the question. If you don't know the answer, say that you "
#             "don't know. Use three sentences maximum and keep the "
#             "answer concise."
#             "\n\n"
#             f"{docs_content}"
#         )
#         conversation_messages = [
#             message
#             for message in state["messages"]
#             if message.type in ("human", "system")
#             or (message.type == "ai" and not message.tool_calls)
#         ]
#         prompt = [SystemMessage(system_message_content)] + conversation_messages

#         # Run
#         response = llm.invoke(prompt)
#         return {"messages": [response]}




#     graph_builder.add_node(query_or_respond)
#     graph_builder.add_node(tools)
#     graph_builder.add_node(generate)

#     graph_builder.set_entry_point("query_or_respond")
#     graph_builder.add_conditional_edges(
#         "query_or_respond",
#         tools_condition,
#         {END: END, "tools": "tools"},
#     )
#     graph_builder.add_edge("tools", "generate")
#     graph_builder.add_edge("generate", END)

#     graph = graph_builder.compile()






#     memory = MemorySaver()
#     graph = graph_builder.compile(checkpointer=memory)

#     # Specify an ID for the thread
#     config = {"configurable": {"thread_id": "abc123"}}



#     result = graph.invoke(
#         {"messages": [{"role": "user", "content": question}]},
#         config=config,
#     )

#     return result["messages"][-1].content 
    