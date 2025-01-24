from langchain_chroma import Chroma
import os
from uuid import uuid4
from fastapi import UploadFile
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config.settings import VECTOR_DB_PATH, OPENAI_API_KEY
from langchain_community.document_loaders import PyPDFLoader

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver




# Initialize Vector Database
vector_store = Chroma(collection_name="file_collection",
                    persist_directory=VECTOR_DB_PATH,
                    embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY))

async def process_and_store_file(file: UploadFile) -> str:
    """r
    Process the uploaded PDF, clear the vector database, and store the new file's embeddings.
    """
    # Generate a unique ID for the file
    # file_id = str(uuid4())
    filename = file.filename

    # Save the uploaded file to a temporary location
    temp_file_path = f"{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await file.read())

    # Load the PDF using LangChain's PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Split text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Clear the vector database before adding new embeddings
    # vector_store.delete_collection()

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)


    # Cleanup the temporary file
    os.remove(temp_file_path)

    return filename





llm = ChatOpenAI(model="gpt-4o-mini")

memory = MemorySaver()


def process_question(filename: str, question: str) -> str:
    """
    Process a user's question and maintain chat history using MemorySaver.
    """

    graph_builder = StateGraph(MessagesState)

    def retrieve_tool_creator(filename: str):
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information of a paper from database."""
            retrieved_docs = vector_store.similarity_search(query, k=10, filter={"source": filename})
            print(retrieved_docs[0])
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        return retrieve

    retrieve = retrieve_tool_creator(filename)


    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])
        system_prompt = (
        "You are an assistant answering questions about a paper attached in the database. "
        "If the question relates to asking about a paper's content, call the `retrieve` tool to fetch context. "
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


    # Use a unique thread ID for the session
    thread_id = filename  # Associate thread ID with the file ID
    config = {"configurable": {"thread_id": thread_id}}

    # Compile the graph with memory for chat history
    graph = graph_builder.compile(checkpointer=memory)

    # Process the question
    result = graph.invoke(
        {"messages": [{"role": "user", "content": f"{question}"}]},
        config=config,
    )

    return result["messages"][-1].content




# file_path = "2409.14879v1.pdf"
# file_path = "fse2024_hassan.pdf"
# loader = PyPDFLoader(file_path)

# docs = loader.load()

# print(len(docs))
# # Split text into chunks for embedding
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(docs)

# # Clear the vector database before adding new embeddings
# # vector_store.delete_collection()
# docs[0]

# uuids = [str(uuid4()) for _ in range(len(docs))]
# vector_store.add_documents(documents=docs, ids=uuids)
# res = vector_store.similarity_search("topic", k=2, filter={"source": file_path})
# res[0]



# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage
# from langgraph.prebuilt import ToolNode
# from langgraph.graph import END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.graph import MessagesState, StateGraph
# from langchain_core.tools import tool
# from langgraph.checkpoint.memory import MemorySaver


# llm = ChatOpenAI(model="gpt-4o-mini")

# memory = MemorySaver()


# filename = "fse2024_hassan.pdf"

# graph_builder = StateGraph(MessagesState)

# @tool(response_format="content_and_artifact")
# def retrieve(query: str, filename: str):
#     """Retrieve information of a paper from database."""
#     print("in retrieve")
#     print("filename passed:", filename)
#     retrieved_docs = vector_store.similarity_search(query, k=2, filter= {"source": filename})
#     print("retrieved_docs: ", retrieved_docs)
#     print(retrieved_docs[0])
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs

# def query_or_respond(state: MessagesState):
#     """Generate tool call for retrieval or respond."""
#     print("in query or respoind")

#     llm_with_tools = llm.bind_tools([retrieve])
#     system_prompt = (
#         "You are an assistant answering questions about a paper attached in the database. "
#         "If the question relates to asking about a paper's content, call the `retrieve` tool to fetch context. "
#         "Otherwise, respond directly."
#     )
#     conversation = [SystemMessage(system_prompt)] + state["messages"]
#     response = llm_with_tools.invoke(conversation)
#     return {"messages": [response]}

# def generate(state: MessagesState):
#     """Generate answer."""

#     print("in generate")
#     recent_tool_messages = [
#         message for message in reversed(state["messages"]) if message.type == "tool"
#     ][::-1]
#     docs_content = "\n\n".join(doc.content for doc in recent_tool_messages)
#     system_message_content = (
#         "You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. If you don't know the answer, say that you "
#         "don't know. Use three sentences maximum and keep the "
#         "answer concise."
#         "\n\n"
#         f"{docs_content}"
#     )
#     prompt = [SystemMessage(system_message_content)] + [
#         message
#         for message in state["messages"]
#         if message.type in ("human", "system")
#         or (message.type == "ai" and not message.tool_calls)
#     ]
#     response = llm.invoke(prompt)
#     return {"messages": [response]}



# graph_builder.add_node(query_or_respond)
# graph_builder.add_node(ToolNode([retrieve]))
# graph_builder.add_node(generate)

# graph_builder.set_entry_point("query_or_respond")
# graph_builder.add_conditional_edges(
#     "query_or_respond",
#     tools_condition,
#     {END: END, "tools": "tools"},
# )
# graph_builder.add_edge("tools", "generate")
# graph_builder.add_edge("generate", END)


# Use a unique thread ID for the session
# thread_id = filename + '6'  # Associate thread ID with the file ID
# config = {"configurable": {"thread_id": thread_id}}

# Compile the graph with memory for chat history
# graph = graph_builder.compile(checkpointer=memory)

# Process the question
# result = graph.invoke(
#     {"messages": [{"role": "user", "content": "what is topic of the paper?"}]},
#     config=config,
# )

# result["messages"][-1].content


# for event in graph.stream({"messages": [{"role": "user", "content": f"what is topic of the paper?, filename:{filename}"}]},
#                            config, stream_mode="values"):
#     event["messages"][-1].pretty_print()


