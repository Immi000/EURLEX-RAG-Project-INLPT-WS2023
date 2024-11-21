import json
from io import StringIO
import requests
import os
import os.path

from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import pandas as pd
from htmlparse import DocumentParser
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from torch.backends.mps import is_available as mps_available
from torch.cuda import is_available as cuda_available
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


INDEX = "nlp-rag-2"

# initialize the environment variables
r = requests.get(
    "https://api.github.com/repos/MaxRumpf/nlpcredentials.github.io/contents/env",
    headers={
        "Authorization": "token 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",
        "Accept": "application/vnd.github.v3.raw",
    },
)
_ = load_dotenv(stream=StringIO(r.text))


def token_len(text):
    return len(tokenizer(text))


# Check if the GPU is available
DEVICE = "cpu"
if mps_available():
    DEVICE = "mps"
elif cuda_available():
    DEVICE = "cuda"
print(f"Using device: {DEVICE}")

if os.path.isfile("doc_links.txt"):
    with open("doc_links.txt", "r") as f:
        doc_links = f.read().splitlines()
    print("Found ", len(doc_links), " documents in file.")
else:
    # Get all the links to the documents
    doc_links = []
    print("Scraping document links...")
    for i in tqdm(range(54)):
        url = f"https://eur-lex.europa.eu/search.html?name=browse-by%3Alegislation-in-force&type=named&displayProfile=allRelAllConsDocProfile&qid=1696858573178&CC_1_CODED=12&page={str(i)}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urlopen(req).read().decode("utf-8")
        soup = BeautifulSoup(html)
        hits = soup.find_all(class_="title")
        for hit in hits:
            doc_links.append(hit.get("name"))
    # store the links in a file in case of failure to prevent re-scraping
    doc_links = list(set(doc_links))
    print("Found ", len(doc_links), " documents")
    with open("doc_links.txt", "w") as f:
        for item in doc_links:
            f.write(f"{item}\n")

# Get the html source for each document and save it
if not os.path.isdir("html"):
    os.mkdir("html")
html_src = []
print("Retrieving html source...")
for link in tqdm(doc_links):
    # skip files that are already downloaded
    if os.path.isfile("html/" + link.split("uri=")[1].replace(":", "-") + ".html"):
        continue

    # obtain link to html source
    req = Request(link, headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(req).read().decode("utf-8")
    soup = BeautifulSoup(html)
    html_src_link = soup.find(id="format_language_table_HTML_EN").get("href")  # type: ignore
    assert type(html_src_link) == str, f"html_src_link is a {type(html_src_link)}"
    safety = 0
    while (
        html_src_link.startswith(".") or html_src_link.startswith("/.")
    ) and safety < 10:
        html_src_link = html_src_link.removeprefix(".")
        html_src_link = html_src_link.removeprefix("/.")
        safety += 1
    html_src_link = "https://eur-lex.europa.eu" + html_src_link
    html_src.append(html_src_link)

    # obtain html source and save
    req = Request(html_src_link, headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(req).read().decode("utf-8")
    soup = BeautifulSoup(html)
    fname = html_src_link.split("uri=")[1].replace(":", "-")
    with open(f"html/{fname}.html", "w") as f:
        f.write(html)


# Parse the html source and save the data in a structured format
with open("html_src_links.txt", "r") as f:
    links = f.readlines()
    links = [link.strip() for link in links]
docs = pd.DataFrame(columns=["doc", "index", "type", "content"])
print("Parsing html source...")
for i, link in tqdm(enumerate(links)):
    fname = link.split("uri=")[-1].replace(":", "-") + ".html"
    try:
        with open("html/" + fname, "r") as f:
            html = f.read()
    except:
        print(f"File {fname} not found.")
        continue
    soup = BeautifulSoup(html, "html.parser")
    if soup.find(class_="alert alert-warning") is not None:
        if (
            "The requested document does not exist."
            in soup.find(class_="alert alert-warning").text
        ):
            print(f"The document {fname} is not available in english.")
            continue
    doc = DocumentParser(soup, link)
    docs = pd.concat([docs, doc.data], ignore_index=True)
docs.to_csv("raw_data.csv", index=False)

# Split the documents into chunks and save it together with metadata
raw = pd.read_csv("raw_data.csv")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf", token=os.getenv("HF_AUTH"), device=DEVICE
)
titles = ["title", "article_no", "subtitle", "table_title", "subsection_no", "header"]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=80,
    length_function=token_len,
    separators=["\n", "\n\n", "  ", " ", ""],
)
idx = 0
docs = {}
print("Splitting the documents...")
for document in tqdm(raw["doc"].unique()):
    for row in raw[raw["doc"] == document].iterrows():
        row = row[1]
        if row["type"] in titles or row["index"] == 0:
            metadata = {"document": document, "title": row["content"]}
        else:
            if len(str(row["content"]).split(" ")) <= 3:
                continue
            chunks = text_splitter.split_text(row["content"])
            for chunk in chunks:
                docs[idx] = [idx, chunk, metadata]
                idx += 1
docs = pd.DataFrame.from_dict(
    docs, orient="index", columns=["index", "chunk", "metadata"]
)
docs.to_csv("docs.csv", index=False)

# Split the documents into chunks and upload to Elasticsearch
client = Elasticsearch(
    cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
    basic_auth=(os.getenv("ELASTIC_USER"), os.getenv("ELASTIC_PASSWORD")),
)
print(client.info())

# Split the documents and embed them
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# embedding_model = 'all-mpnet-base-v2'
emb_model = HuggingFaceEmbeddings(
    model_name=embedding_model, model_kwargs={"device": DEVICE}, show_progress=True
)
docs = pd.read_csv("docs.csv")
with open("dates.json") as f:
    dates = json.load(f)
documents = []
chunks = docs["chunk"].tolist()
print("Embedding the data...")
embeddings = emb_model.embed_documents(chunks)
for i, (_, row) in tqdm(enumerate(docs.iterrows())):
    # id = row['index']
    metadata = eval(row["metadata"].replace("'title': nan", "'title': ''"))
    title = metadata["title"]
    doc_id = metadata["document"]
    doc = {}
    doc["title"] = title
    doc["doc_id"] = doc_id
    doc["text"] = row["chunk"]
    doc["embedding"] = embeddings[i]
    doc["date"] = dates[doc_id]
    documents.append(doc)

# Upload the data to the Elasticsearch index
print("Uploading data to Elasticsearch...")
actions_list = []
for i in tqdm(range(len(documents))):
    doc = documents[i]
    actions_list.append({"index": INDEX, "_id": str(i), "doc": doc})
    if i % 1000 == 0:
        helpers.bulk(client=client, actions=actions_list, index=INDEX)
        actions_list = []
helpers.bulk(client=client, actions=actions_list, index=INDEX)
