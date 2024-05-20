import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FILEPATH = "book/Emma_by_Jane_Austen.txt"


QUESTION1 = "Why does Emma befriend Harriet?" 
QUESTION2 = "How does Harriet benefit from Emma's friendship?"
QUESTION3 = "Why isn't Harriet a good companion for Emma?"
QUESTION4 = "Why doesn't Emma befriend Jane Fairfax?"



# 1. Create a Vector Store from text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def create_index_from_text(path: str, openaikey: str) -> Chroma:

    file = open(path, 'rb')
    bytes = file.read()
    content = bytes.decode("utf-8")
    file.close()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    documents = text_splitter.create_documents([content])

    embeddings = OpenAIEmbeddings(openai_api_key=openaikey)
    index = Chroma.from_documents(documents=documents, embedding=embeddings)
    print("Index created!")

    return index




# 2. Retrieval Augmented Generation (RAG) and Query ChatGPT
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
def generate_answer(query: str, index: Chroma, openaikey: str) -> str:

    # Get relevant data with similarity search
    texts = index.similarity_search(query)

    # Generate answer with OpenAI
    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613",openai_api_key=openaikey)
    template = """
    CONTEXT: {docs}
    You are a helpful assistant, above is some context, 
    Please answer the question, and make sure you follow ALL of the rules below:
    1. Answer the questions only based on context provided, do not make things up
    2. Answer questions in a helpful manner that straight to the point, with clear structure & all relevant information that might help users answer the question
    3. Anwser should be formatted in Markdown
    4. If there are relevant images, video, links, they are very important reference data, please include them as part of the answer

    QUESTION: {query}
    ANSWER (formatted in Markdown):
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    response = chain.invoke({"docs": texts, "query": query})

    return response.content




def main():
    print("Starting...")
    chroma_index = create_index_from_text(FILEPATH, OPENAI_API_KEY)

    questions = [QUESTION1, QUESTION2, QUESTION3, QUESTION4]

    print("Creating Responses...")
    print("========")
    for question in questions:
        response = generate_answer(question, chroma_index, OPENAI_API_KEY)
        print(question)
        print(response)
        print(" ")



if __name__ == "__main__":
    main()