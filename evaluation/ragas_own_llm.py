import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,6,7,8,9" # important to set before imports using torch

from ragas import evaluate, llms
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
)
from datasets import Dataset

# from chatmodel import RAG
from tqdm.auto import tqdm
import pandas as pd

# from ragas.llms.base import BaseRagasLLM
# from ragas.embeddings.base import BaseRagasEmbeddings
from llm import HuggingfaceLlama2, ChatGPT

# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from transformers import pipeline
# from transformers import LlamaTokenizer
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from io import StringIO
import requests

r = requests.get(
    "https://api.github.com/repos/MaxRumpf/nlpcredentials.github.io/contents/env",
    headers={
        "Authorization": "token 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",
        "Accept": "application/vnd.github.v3.raw",
    },
)
_ = load_dotenv(stream=StringIO(r.text))
overall_result_values = None
for i in range(0, 40):
    dataset_pd = pd.read_pickle("new_embeddingresults.pkl")
    dataset_pd = dataset_pd.rename(
        columns={
            "questions": "question",
            "context": "contexts",
            "sample_answers": "answer",
            "model_answers": "ground_truth",
        }
    )
    if len(dataset_pd.iloc[i]["contexts"]) == 0:
        dataset_pd.iloc[i]["contexts"] = [""]
    dataset = Dataset.from_pandas(dataset_pd.iloc[i : i + 1])
    if dataset["question"][0].startswith("What role can private capital play in"):
        os.exit(0)
    print("dataset", dataset)
    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            context_relevancy,
            answer_correctness,
        ],
        raise_exceptions=False,
        # max_workers=4,
        llm=llms.llm_factory(model="gpt-3.5-turbo-0125"),
    )
    result_pandas = result.to_pandas()
    if overall_result_values is None:
        overall_result_values = result_pandas
    else:
        overall_result_values = pd.concat(
            [overall_result_values, result_pandas], ignore_index=True
        )
    overall_result_values.to_pickle("new_embeddingresults_ragas_firsthalf.pkl")
    print(overall_result_values)
