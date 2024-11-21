import os
import torch
from elasticsearch import Elasticsearch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from requests.exceptions import ConnectTimeout


class DatabaseConnection:
    """
    Client for interfacing with the Elasticsearch database. Provides methods for searching and enriching search results.
    Requires the environment variables ELASTIC_USER, ELASTIC_PASSWORD to be set. ELASTIC_CLOUD_ID is the cloud ID
    of the Elasticsearch instance and ELASTIC_PASSWORD is the password for the elasticsearch user ELASTIC_USER.

    :param verbose: Whether to print debug information, defaults to True
    :type verbose: bool, optional
    :param embedding_model: The name of the HuggingFace model to use for embeddings, defaults to "sentence-transformers/all-MiniLM-L6-v2"
    :type embedding_model: str, optional
    :param index_name: The name of the Elasticsearch index to use, defaults to "nlp-rag"
    :type index_name: str, optional
    :param embedding_device: The device to use for embeddings, defaults to "auto". Other options are "cpu", "cuda", and "mps".
    :type embedding_device: str, optional
    """

    def __init__(
        self,
        verbose: bool = True,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_name: str = "nlp-rag-large-date-2",
        embedding_device: str = "auto",
    ):
        self.verbose = verbose
        self.index_name = index_name
        if embedding_device == "auto":
            self.emb_device = "cpu"
            if torch.backends.mps.is_available():
                self.emb_device = "mps"
            if torch.cuda.is_available():
                self.emb_device = "cuda"
        else:
            self.emb_device = embedding_device
        self._print("Embedding device:", self.emb_device)
        if os.getenv("ELASTIC_CLOUD_ID") is None:
            raise ValueError("The environment variable ELASTIC_CLOUD_ID is not set.")
        if os.getenv("ELASTIC_PASSWORD") is None:
            raise ValueError("The environment variable ELASTIC_PASSWORD is not set.")

        self.client = Elasticsearch(
            cloud_id=os.getenv("ELASTIC_CLOUD_ID"),
            basic_auth=(os.getenv("ELASTIC_USER"), os.getenv("ELASTIC_PASSWORD")),
        )  # type: ignore
        self.emb_model = HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs={"device": self.emb_device}
        )
        self.emb_model_name = embedding_model

    def update_embedding_model(self, emb_model):
        if self.emb_model_name != emb_model:
            self._print("received updated embedding model", emb_model)
            self.emb_model = HuggingFaceEmbeddings(
                model_name=emb_model, model_kwargs={"device": self.emb_device}
            )
            self.emb_model_name = emb_model

    def enrich_with_neighboring_chunks(
        self, matches: list[dict], num_neighbors: int = 1
    ):
        """
        Enriches search results with the text of neighboring chunks by retrieving the database entries with neighboring IDs
        and merging the text (or concatenating in case of no overlap) into the search results.

        :param matches: The search results to enrich.
        :type matches: list[dict]
        :param num_neighbors: The number of neighboring chunks to enrich with, defaults to 1
        :type num_neighbors: int, optional
        :return: The same search results as the input but with the text of neighboring chunks added to the "text" field.
        :rtype: list[dict]
        """
        if num_neighbors < 1:
            return matches
        enriched_matches = []
        if self.index_name == "only_recursive_splitter-2":
            for match in matches:
                for i in range(1, num_neighbors + 1):
                    doc_celex = match["id"].split("-")[0]
                    doc_id = int(match["id"].split("-")[1])
                    results = self.client.search(
                        index=self.index_name,
                        query={
                            "terms": {
                                "_id": [
                                    doc_celex + "-" + str(doc_id - i),
                                    doc_celex + "-" + str(doc_id - i),
                                ]
                            }
                        },
                    )
                    processed = self._process_results(results)
                    enriched_matches.extend(processed)
                enriched_matches.append(match)
        else:
            for match in matches:
                for i in range(1, num_neighbors + 1):
                    results = self.client.search(
                        index=self.index_name,
                        query={
                            "terms": {
                                "_id": [int(match["id"]) - i, int(match["id"]) + i]
                            }
                        },
                    )
                    # TODO: Handle case if retrieved chunk +/- num_neighbors goes to previous or next document,
                    # or does not even exist (think first or last document)
                    # -> Very rare, for now apply the Ostrich algorithm
                    processed = self._process_results(results)
                    match["text"] = self._merge_strings(
                        processed[0]["text"], match["text"]
                    )
                    match["text"] = self._merge_strings(
                        match["text"], processed[1]["text"]
                    )
                enriched_matches.append(match)
        return enriched_matches

    def similarity_search(
        self, search_query: str, top_k: int = 10, avoid_duplicates: bool = False
    ):
        """
        Calls the elasticsearch client to perform a similarity search with the given query. The query is being embedded
        and used to search for similar texts in the database and the top k results are returned. Optionally filters out
        duplicate texts.

        :param query: The query to search for.
        :type query: str
        :param top_k: The number of results to return, defaults to 10
        :type top_k: int, optional
        :param avoid_duplicates: Whether to avoid returning duplicate texts, defaults to False
        :type avoid_duplicates: bool, optional
        :return: The search results.
        :rtype: list[dict]
        """
        query_emb = self.emb_model.embed_query(search_query)
        if avoid_duplicates:
            results = self.client.search(
                index=self.index_name,
                size=top_k,
                query={
                    "size": top_k,
                    "knn": {
                        "field": "doc.embedding",
                        "query_vector": query_emb,
                        "num_candidates": top_k,
                    },
                },
                collapse={"field": "doc.text.keyword"},
                aggs={
                    "unique_texts": {
                        "terms": {"field": "doc.text.keyword", "size": 100000}
                    }
                },
            )
        else:
            results = self.client.search(
                index=self.index_name,
                size=top_k,
                query={
                    "knn": {
                        "field": "doc.embedding",
                        "query_vector": query_emb,
                        "num_candidates": top_k,
                    },
                },
            )
        proc = self._process_results(results)
        self._print("Similarity search:")
        self._print("Query:", search_query)
        self._print("Results:", proc)
        return proc

    def hybrid_search(
        self,
        search_query: str,
        keywords: list[str],
        top_k: int = 10,
        date_range: list[dict] = [],
    ):
        """
        Calls the elasticsearch client to perform a hybrid search with the given query, keywords, and date range.
        The query is being embedded and used for similarity search, while the keywords are used for keyword search.
        The date range is used for filtering the results. The results of the similarity and keyword search are then
        merged using reciprocal rank fusion (RRF) and returned.

        :param search_query: The query to search for with similarity search.
        :type search_query: str
        :param keywords: The keywords to search for with keyword search. Must be a list of keywords, e.g. ["keyword1", "keyword2"].
        :type keywords: list[str]
        :param top_k: The number of results to return, defaults to 10
        :type top_k: int, optional
        :param date_range: The date range to filter the results with, defaults to []. Must be a list of dictionaries
            with keys "start_date" and "end_date", e.g. [{"start_date": "01.01.2000", "end_date": "31.12.2020"}]. Only one of the
            two dates is required, the other can be set to None. The date format must be "dd.mm.yyyy".
        :type date_range: list[dict], optional
        :return: The search results.
        :rtype: list[dict]
        """
        query_emb = self.emb_model.embed_query(search_query)
        should = []
        for d_range in date_range:
            obj = {"bool": {"filter": [{"range": {"doc.date": {}}}]}}
            if "start_date" in d_range.keys() and d_range["start_date"] is not None:
                obj["bool"]["filter"][0]["range"]["doc.date"]["gte"] = d_range[
                    "start_date"
                ]
            if "end_date" in d_range.keys() and d_range["end_date"] is not None:
                obj["bool"]["filter"][0]["range"]["doc.date"]["lte"] = d_range[
                    "end_date"
                ]
            should.append(obj)
        knn = {
            "field": "doc.embedding",
            "query_vector": query_emb,
            "k": top_k,
            "num_candidates": 2 * top_k,
            "filter": should,
        }
        if len(should) < 1:
            del knn["filter"]
        query = {
            "bool": {"must": {"match": {"doc.text": " ".join(keywords)}}, "filter": []}
        }
        if len(should) > 0:
            query["bool"]["filter"].append(
                {"bool": {"should": should, "minimum_should_match": 1}}
            )
        results = self.client.search(
            index=self.index_name,
            size=top_k,
            query=query,
            knn=knn,
            rank={
                "rrf": {
                    "window_size": 500,
                    # "rank_constant": 20
                }
            },
        )

        proc = self._process_results(results)
        self._print("Hybrid search:")
        self._print("Query:", query)
        self._print("Results:", proc)
        return proc

    def _process_results(self, results):
        """
        Processes the search results from the elasticsearch client and returns them as a list of dictionaries.
        The dictionaries are formatted as follows:
        {
            "id": str,
            "score": float,
            "title": str,
            "document": str,
            "text": str,
            "date": str
        }

        :param results: The search results to process. Should be in the format returned by the elasticsearch client.
        :type results: dict
        :return: The processed search results.
        :rtype: list[dict]
        """
        if results.body["timed_out"] == True:
            raise TimeoutError(
                "The connection to the elasticsearch database timed out, please try again."
            )
        matches = []
        for hit in results.body["hits"]["hits"]:
            match = {}
            match["id"] = hit["_id"]
            match["score"] = hit["_score"]
            match["title"] = hit["_source"]["doc"]["title"]
            match["document"] = hit["_source"]["doc"]["doc_id"]
            match["text"] = hit["_source"]["doc"]["text"]
            if "date" in hit["_source"]["doc"]:
                match["date"] = hit["_source"]["doc"]["date"]
            matches.append(match)
        return matches

    def _merge_strings(self, s1: str, s2: str) -> str:
        """
        Helper function that merges two strings that have some overlap by removing the overlapping part from the first
        string and concatenating the two strings. If there is no overlap, the two strings are simply concatenated.
        Only checks if the second string starts with the first string, not the other way around.

        :param s1: The first string to merge.
        :type s1: str
        :param s2: The second string to merge.
        :type s2: str
        :return: The merged string.
        :rtype: str
        """
        for i in range(len(s1)):
            if s2.startswith(s1[i:]):
                return s1[:i] + s2
        return s1 + " " + s2

    def _print(self, *args):
        """
        Prints the arguments if the verbose flag is set to True. Otherwise does nothing.
        """
        if self.verbose:
            print(*args)
