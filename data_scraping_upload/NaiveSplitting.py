import os
from dotenv import load_dotenv, find_dotenv
from io import StringIO
import requests

##load environment and access keys
_ = load_dotenv()


##load Llama Tokenizer for embedding
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf", use_auth_token=os.getenv("HF_AUTH")
)


def token_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


# import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# instantiate and configure RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    length_function=token_len,
    separators=["\n\n", "\n", " ", ""],
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os

# load and instantiate Embedding Model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda:0"  # make sure you are on gpu
### your code ###
embed_model = HuggingFaceEmbeddings(
    model_name=embedding_model, model_kwargs={}, encode_kwargs={"batch_size": 32}
)

# Set up connection to Elasticsearch and create index
from elasticsearch import Elasticsearch, helpers

client = Elasticsearch(
    cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
    basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")),
)
client.info()
client.indices.create(index="only_recursive_splitter")

import json

# read dates.json file which contains mapping for document CELEX ID -> date
with open("dates.json", "r") as f:
    dates_dics = json.load(f)
print(dates_dics)

import os.path
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
import urllib.request

MISSING_DATES = []


# function to extract documents from page on EUR-Lex Search
def download_pdfs(soup):
    print("found ", len(soup.find_all("a", {"class": "title"})), "items on page")
    for a in soup.find_all("a", {"class": "title"}):
        celex = a["href"].replace("./legal-content/AUTO/?uri=", "")
        celex = celex[: celex.find("&qid=")]
        pdf_href = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=" + celex
        pdf_path = "docs/" + celex + ".pdf"
        # print(celex)
        try:
            urllib.request.urlretrieve(pdf_href, pdf_path)
        except:
            print("could not retrieve ", pdf_href)
            continue
        index_pdf(pdf_path, celex)
        os.remove(pdf_path)
    if len(soup.find_all("a", {"class": "title"})) < 10:
        print("length < 10, will exit")
        os.exit(0)


# Function to download PDF Version of each document
def download_page(page_num):
    URL = (
        "https://eur-lex.europa.eu/search.html?name=browse-by%3Alegislation-in-force&type=named&displayProfile=allRelAllConsDocProfile&qid=1696858573178&CC_1_CODED=12&page="
        + str(page_num)
    )
    print("retrieving page ", page_num)
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, "html.parser")
    download_pdfs(soup)


# Read PDF Document, embed using HuggingFaceEmbeddings and upload to Elastis
from langchain.document_loaders import UnstructuredPDFLoader


def index_pdf(path, celex_id):
    loader = UnstructuredPDFLoader(path)
    try:
        data = loader.load()[0]
    except:
        print("could not load", celex_id)
        return
    chunks = text_splitter.split_text(data.page_content)
    documents = []
    embeddings = embed_model.embed_documents(chunks)
    for i, chunk in enumerate(chunks):
        doc = {}
        doc["doc_id"] = f"{celex_id}-{i}"
        doc["text"] = chunk
        doc["title"] = celex_id
        doc["embedding"] = embeddings[i]
        celexid = celex_id.replace(":", "_")
        if celexid in dates_dics.keys():
            doc["date"] = dates_dics[celexid]
        else:
            if celexid not in MISSING_DATES:
                MISSING_DATES.append(celexid)
                print("MISSING_DATES", MISSING_DATES)
            continue
        # print("Date for ", celexid, "is ", dates_dics[celexid])
        documents.append(doc)
    actions_list = []
    for i, doc in enumerate(documents):
        actions_list.append(
            {"index": "only_recursive_splitter", "_id": doc["doc_id"], "doc": doc}
        )
        if i % 100 == 0:
            helpers.bulk(
                client=client, actions=actions_list, index="only_recursive_splitter"
            )
            actions_list = []
    helpers.bulk(client=client, actions=actions_list, index="only_recursive_splitter")
    actions_list = []


# Download Content on Search-Pages in loop
for i in range(0, 100):
    download_page(i)
    f = open("pages.txt", "w")
    f.write("Page " + str(i))
    f.close()
