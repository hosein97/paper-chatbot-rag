from langchain_chroma import Chroma
import os
from uuid import uuid4
from fastapi import UploadFile
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config.settings import VECTOR_DB_PATH, OPENAI_API_KEY
from langchain_community.document_loaders import PyPDFLoader



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
    temp_file_path = f"/tmp/{file.filename}"
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


# import os
# from langchain_core.vectorstores import InMemoryVectorStore
# VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./app/vector_db")  # Path to store the vector DB
# vector_store = Chroma(collection_name="file_collection",
#                       persist_directory=VECTOR_DB_PATH,
#                     embedding_function=OpenAIEmbeddings())

# vector_store.delete_collection()

# uuids = [str(uuid4()) for _ in range(len(all_splits))]
# vector_store.add_documents(documents=all_splits, ids=uuids)



# from langchain.document_loaders import PyPDFLoader


# file_path = "2409.14879v1.pdf"
# loader = PyPDFLoader(file_path)

# docs = loader.load()

# # print(len(docs))

# # from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200, add_start_index=True
# )
# all_splits = text_splitter.split_documents(docs)
# all_splits[0]
# # len(all_splits)

# retrieved_docs = vector_store.similarity_search("contribution of the paper?", k=2, filter={"source": file_path})
# retrieved_docs[0]
# # all_splits[0].metadata


# from app.config.settings import VECTOR_DB_PATH
# from langchain_chroma import Chroma
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




# vector_store = Chroma(persist_directory=VECTOR_DB_PATH,
#                         collection_name="file_collection",
#                         )

# graph_builder = StateGraph(MessagesState)

# @tool(response_format="content_and_artifact")
# def retrieve(query: str):
#     """Retrieve information from a paper."""
#     retrieved_docs = vector_store.similarity_search(query, k=2)
#     print(retrieved_docs[0])
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs

# def query_or_respond(state: MessagesState):
#     """Generate tool call for retrieval or respond."""
#     llm_with_tools = llm.bind_tools([retrieve])
#     system_prompt = (
#         "You are an assistant answering questions about a paper stored in a database. "
#         "If the question relates to the paper's content, call the `retrieve` tool to fetch context. "
#         "Otherwise, respond directly."
#     )
#     conversation = [SystemMessage(system_prompt)] + state["messages"]
#     response = llm_with_tools.invoke(conversation)
#     return {"messages": [response]}

# def generate(state: MessagesState):
#     """Generate answer."""
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

# # Compile the graph with memory for chat history
# graph = graph_builder.compile(checkpointer=memory)

# # Use a unique thread ID for the session
# thread_id = "2409.14879v1.pdf"  # Associate thread ID with the file ID
# config = {"configurable": {"thread_id": thread_id}}

# # Process the question
# result = graph.invoke(
#     {"messages": [{"role": "user", "content": "Contribution of the paper?"}]},
#     config=config,
# )

# result["messages"][-1].content

# for step in graph.stream(
#     {"messages": [{"role": "user", "content": "Contribution of the paper?"}]},
#     stream_mode="values",
#     config=config,
# ):
#     step["messages"][-1].pretty_print()















# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o-mini")


# from langchain.embeddings import OpenAIEmbeddings


# from langchain_openai import OpenAIEmbeddings

# from dotenv import load_dotenv

# load_dotenv()

# file_id = str(uuid.uuid4())

# metadata = {"file_id": file_id, "file_name": file_path}
# all_splits[0]
# vector_store.add_documents(documents=all_splits)

# from langgraph.graph import MessagesState, StateGraph

# graph_builder = StateGraph(MessagesState)


# from langchain_core.tools import tool


# @tool(response_format="content_and_artifact")
# def retrieve(query: str):
#     """Retrieve information of the paper."""
#     retrieved_docs = vector_store.similarity_search(query, k=2)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs

# from langchain_core.messages import SystemMessage
# from langgraph.prebuilt import ToolNode


# # Step 1: Generate an AIMessage that may include a tool-call to be sent.
# def query_or_respond(state: MessagesState):
#     """Generate tool call for retrieval or respond."""
#     llm_with_tools = llm.bind_tools([retrieve])
#     # response = llm_with_tools.invoke(state["messages"])
#     # # MessagesState appends messages to state instead of overwriting
#     # return {"messages": [response]}
#     system_prompt = (
#         "You are an assistant answering questions about a paper stored in a database. "
#         "If the question relates to the paper's content, call the `retrieve` tool to fetch context. "
#         "Otherwise, respond directly."
#     )
#     conversation = [
#         SystemMessage(system_prompt)
#     ] + state["messages"]

#     response = llm_with_tools.invoke(conversation)
#     print("Generated response:", response)  # Debugging
#     return {"messages": [response]}



# # Step 2: Execute the retrieval.
# tools = ToolNode([retrieve])


# # Step 3: Generate a response using the retrieved content.
# def generate(state: MessagesState):
#     """Generate answer."""
#     # Get generated ToolMessages
#     recent_tool_messages = []
#     for message in reversed(state["messages"]):
#         if message.type == "tool":
#             recent_tool_messages.append(message)
#         else:
#             break
#     tool_messages = recent_tool_messages[::-1]

#     # Format into prompt
#     docs_content = "\n\n".join(doc.content for doc in tool_messages)
#     system_message_content = (
#         "You are an assistant for question-answering task. "
#         "Use the following pieces of retrieved context to answer "
#         "the question. If you don't know the answer, say that you "
#         "don't know. Use three sentences maximum and keep the "
#         "answer concise."
#         "\n\n"
#         f"{docs_content}"
#     )
#     conversation_messages = [
#         message
#         for message in state["messages"]
#         if message.type in ("human", "system")
#         or (message.type == "ai" and not message.tool_calls)
#     ]
#     prompt = [SystemMessage(system_message_content)] + conversation_messages

#     # Run
#     response = llm.invoke(prompt)
#     return {"messages": [response]}



# from langgraph.graph import END
# from langgraph.prebuilt import ToolNode, tools_condition

# graph_builder.add_node(query_or_respond)
# graph_builder.add_node(tools)
# graph_builder.add_node(generate)

# graph_builder.set_entry_point("query_or_respond")
# graph_builder.add_conditional_edges(
#     "query_or_respond",
#     tools_condition,
#     {END: END, "tools": "tools"},
# )
# graph_builder.add_edge("tools", "generate")
# graph_builder.add_edge("generate", END)

# graph = graph_builder.compile()



# input_message = "What the topic of the paper?"

# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

# from langgraph.checkpoint.memory import MemorySaver

# memory = MemorySaver()
# memory3 = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)
# graph2 = graph_builder.compile(checkpointer=memory)
# graph3 = graph_builder.compile(checkpointer=memory3)

# # Specify an ID for the thread
# config = {"configurable": {"thread_id": "abc123"}}


# input_message = "What was the main contribution of the paper?"

# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
# ):
#     step["messages"][-1].pretty_print()



# input_message = "Can you reprase first sentence of what you told before?"

# for step in graph.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
# ):
#     step["messages"][-1].pretty_print()



# result = graph.invoke(
#     {"messages": [{"role": "user", "content": input_message}]},
#     config=config,
# )

# result = graph3.invoke(
#     {"messages": [{"role": "user", "content": input_message}]},
#     config=config,
# )


# result["messages"][-1].content


