import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma

from colorama import Fore

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


load_dotenv()

DATABRICKS_API_KEY = os.environ["DATABRICKS_API_KEY"]
DATABRICKS_BASE_URL = os.environ["DATABRICKS_BASE_URL"]

llm = ChatOpenAI(
    api_key=DATABRICKS_API_KEY,
    base_url=DATABRICKS_BASE_URL,
    model="databricks-gpt-oss-20b",
)

# Databricks serving endpoints are OpenAI-compatible, so OpenAIEmbeddings
# works if we point it at the same base URL. `check_embedding_ctx_length=False`
# disables tiktoken lookup for the (non-OpenAI) model name.
embeddings = OpenAIEmbeddings(
    api_key=DATABRICKS_API_KEY,
    base_url=DATABRICKS_BASE_URL,
    model="databricks-gte-large-en",
    check_embedding_ctx_length=False,
)

documents = [
    "Harrison worked at Databricks for 10 years.",
    "Databricks is a data and AI company founded in 2013.",
    "Harrison built vector search systems.",
]

vector_store = FAISS.from_texts(documents, embedding=embeddings)

prompt_template = PromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {question}
Answer:"""
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Answer question
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

chat_history = []  # Collect chat history here (a sequence of messages)


def _content_to_str(content) -> str:
    """Flatten AIMessage.content (str or list of blocks) to a plain string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text") or block.get("content") or "")
    return "".join(parts)

def query(question: str, k: int = 2):
    # docs = vector_store.similarity_search(question, k=k)
    # retriever = vector_store.as_retriever(search_kwargs={"k": k})
    # docs = retriever.invoke(question, top_k=k)

    documents = load_split_documents()
    retriever = load_embeddings(documents, question)

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    # retrieval_chain = (
    #     {
    #         "context": retriever,
    #         "question": RunnablePassthrough(),
    #     }
    #     | prompt
    #     | llm
    #     | StrOutputParser()

    # )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Usage:
    response = rag_chain.invoke({"input": question, "chat_history": chat_history})

    # Persist this turn so follow-up questions can reference it.
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response["answer"]))

    # response = retrieval_chain.invoke(question)
    print(response)

    # print(docs)
    # context = "\n".join(d.page_content for d in docs)
    # prompt = prompt_template.format(context=context, question=question)
    return response


def answer(question: str, k: int = 2) -> str:
    query(question=question)["answer"]

def load_embeddings(documents, user_query):
    """Create a vector store from a set of documents."""
    db = Chroma.from_documents(documents, embeddings)
    docs = db.similarity_search(user_query)
    print(docs)
    return db.as_retriever()


def load_split_documents():
    """Load a file from path, split it into chunks, embed each chunk and load it into the vector store."""
    raw_text = TextLoader("./docs/faq.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=".")
    chunks = text_splitter.split_documents(raw_text)
    print(f"number of chunks {len(chunks)}")
    print(chunks[0])
    return chunks


def start():
    instructions = (
        """Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"""
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        start()


def ask():
    while True:
        user_input = input("Q: ")
        # Exit
        if user_input == "x":
            start()
        else:

            response = answer(user_input)

            print(Fore.BLUE + "A: " + response + Fore.RESET)
            print(Fore.WHITE + 
                  "\n-------------------------------------------------")


if __name__ == "__main__":
    start()



# if __name__ == "__main__":
#     query = "what's return policy in Renshou?"
#     print(Fore.CYAN + f"Q: {query}" + Fore.RESET)
#     print(Fore.BLUE + f"A: {answer(query)}" + Fore.RESET)
