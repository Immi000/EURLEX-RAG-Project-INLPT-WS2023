import os
import ast
from dotenv import load_dotenv
from llm import ChatGPT, OllamaModel, HuggingfaceLlama2, Model
from db_api import DatabaseConnection
from io import StringIO
import requests

INDEX_MAP = {
    "Semantic-Splitting2N": "nlp-rag-2",
    "NaiveSplitting": "only_recursive_splitter-2",
    "Semantic-Splitting-1N": "nlp-rag-2",
    "Semantic-Splitting2N-LE": "nlp-rag-large-date-2",
}


class RAG:
    """
    Main class for the RAG model. It uses the DatabaseConnection class to connect to the ElasticSearch database and one of the LLM classes
    to generate the answer. The class also contains methods to improve the question by detecting follow up questions, extracting keywords,
    enriching the retrieved documents with their neighbors and detecting time dependence to filter the search.

    :param embedding_model: The name of the embedding model to use. Default is "sentence-transformers/all-MiniLM-L6-v2".
    :type embedding_model: str
    :param index_name: The name of the ElasticSearch index to use. Default is "nlp-rag-reindexed".
    :type index_name: str
    :param embedding_device: The device to use for the embedding model. Default is "auto".
    :type embedding_device: str
    :param verbose: Whether to print logs during the processing. Default is True.
    :type verbose: bool
    :param use_results: The number of neighboring chunks to retrieve. Default is 10.
    :type use_results: int
    :param context_enrichment: The number of neighboring chunks to retrieve. Default is 1.
    :type context_enrichment: int
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_name: str = "only_recursive_splitter-2",
        embedding_device: str = "auto",
        verbose: bool = True,
        use_results: int = 10,
        context_enrichment: int = 2,
    ):
        self._load_environ()
        self.use_results = use_results
        self.context_enrichment = context_enrichment
        self.index_name = index_name
        self.verbose = verbose
        self.db_api = DatabaseConnection(
            verbose=verbose,
            embedding_model=embedding_model,
            index_name=index_name,
            embedding_device=embedding_device,
        )
        self.model = ChatGPT()
        self.chat_history = []
        self.previous_context = None

    def set_index(self, index_name):
        if index_name not in INDEX_MAP.keys():
            raise ValueError("Unknown index!")
        self._print("Received new index choice:", index_name)
        self.index_name = INDEX_MAP[index_name]
        self.db_api.index_name = INDEX_MAP[index_name]
        if index_name in [
            "Semantic-Splitting-1N",
            "Semantic-Splitting2N",
            "NaiveSplitting",
        ]:
            self.db_api.update_embedding_model("sentence-transformers/all-MiniLM-L6-v2")
        elif index_name == "Semantic-Splitting2N-LE":
            self.db_api.update_embedding_model("all-mpnet-base-v2")
        else:
            raise ValueError("Unknown index!")
        if index_name == "Semantic-Splitting-1N" or index_name == "NaiveSplitting":
            self.context_enrichment = 1
        else:
            self.context_enrichment = 2
        self._print("Updated context enrichment to:", self.context_enrichment)

    def _load_environ(self):
        """
        Method to load the environment variables. The retrieved variables are
        ELASTIC_CLOUD_ID, ELASTIC_USER, ELASTIC_PASSWORD, OPENAI_API_KEY.
        """

        load_dotenv()
        if len(os.environ["ELASTIC_CLOUD_ID"]) < 5:
            raise ValueError("Environment variable ELASTIC_CLOUD_ID not valid!")
        if len(os.environ["ELASTIC_USER"]) < 5:
            raise ValueError("Environment variable ELASTIC_USER not valid!")
        if len(os.environ["ELASTIC_PASSWORD"]) < 2:
            raise ValueError("Environment variable ELASTIC_PASSWORD not valid!")
        if len(os.environ["OPENAI_API_KEY"]) < 5:
            raise ValueError("Environment variable OPENAI_API_KEY not valid!")

    def _generate_context_line(self, cont: dict) -> str:
        """
        This method generates a line of context from the retrieved documents in the format that is used in the prompt for the language model.
        It writes down the document of origin, the title, the context text and the date of the document if available. It also generates a URL
        to the document from the CELEX ID of the document.

        :param cont: The context dictionary to generate the line from.
        :type cont: dict
        :return: The generated line of context in text form.
        :rtype: str
        """
        celex_id = cont["document"].split("-")[0].replace("_", ":")
        url = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=" + celex_id
        d = cont["date"] if "date" in cont else ""
        return (
            f"\nSource: {celex_id}, Title: {cont['title']}, URL: {url}, {d}\nContext: {cont['text']}\n"
            ""
        )

    def ask_question(
        self,
        question: str,
        improve_follow_up: bool = True,
        use_keywords: bool = True,
        detect_time_dependence: bool = True,
    ) -> str:
        """
        Method to ask a question to the RAG model. It retrieves relevant context from the database and formulates a prompt for the language model
        to generate an answer. Optionally, the question can be improved by detecting whether it is a follow-up question and reformulating it to
        include the important information from the previous question. It can also optionally extract keywords and generate topically adjacent keywords
        from the question to use in a hybrid search. Additionally, it can detect time dependence in the question and filter the search results by
        the date of the documents.

        :param question: The question to ask the RAG model.
        :type question: str
        :param improve_follow_up: Whether to improve the question by detecting follow-up questions and reformulating them. Default is True.
        :type improve_follow_up: bool
        :param use_keywords: Whether to use keywords from the question to perform a hybrid search. Default is True. If False, performs a similarity search instead.
        :type use_keywords: bool
        :param detect_time_dependence: Whether to detect time dependence in the question and filter the search results by the date of the documents. Default is True.
        :type detect_time_dependence: bool
        :return: The generated answer from the language model.
        :rtype: str
        """
        if not isinstance(question, str):
            raise ValueError("Question must be a string.")
        if len(self.chat_history) > 0 and improve_follow_up:
            question = self._improve_follow_up_question(
                self.chat_history[-1]["question"], question
            )
        if detect_time_dependence:
            date_range = self._detect_time_dependence(question)
        else:
            date_range = []
        if use_keywords:
            keywords = self._extract_keywords(question)
            self.context = self.db_api.hybrid_search(
                question,
                keywords=keywords,
                top_k=self.use_results,
                date_range=date_range,
            )
        else:
            self.context = self.db_api.similarity_search(
                question, top_k=self.use_results
            )
        if self.context_enrichment > 0:
            self.context = self.db_api.enrich_with_neighboring_chunks(
                self.context, num_neighbors=self.context_enrichment
            )
        prompt = ""
        self._add_previous_context(prompt)
        prompt += "Information from the EURLEX Database on energy law:\n---------------------\n"
        for cont in self.context:
            prompt += self._generate_context_line(cont)
        prompt += f"\n---------------------\n"
        self._add_previous_chat_history(prompt)
        prompt += "\n"
        prompt += f"\nGiven the provided information from the EURLEX database on energy law above, answer the "
        prompt += "following question using ONLY THE PROVIDED INFORMATION FROM THE EURLEX DATABASE.\nIf the question "
        prompt += "relates to a previously asked question, also consider the previous context for your answer.\nCite "
        prompt += "the given sources where applicable. Format your answer using markdown and use it to highlight "
        prompt += "important aspects! Do not engage on topics that are not related to the European law.\n"
        prompt += "Whenever you are referencing any given documents, rephrase the part of the information from the "
        prompt += "EURLEX database on energy law that is important for the answer and cite the source together with "
        prompt += "its ID and URL in the following format: [CELEX ID](URL).\n"
        prompt += f"\nQuestion:\n{question}\nHelpful answer:\n"
        self._print(prompt)
        self.previous_context = self.context
        self.answer = self.model(prompt)
        self._print(self.answer)
        hist = {"question": question, "answer": self.answer}
        self.chat_history.append(hist)
        return self.answer

    def get_context(self):
        context = []
        for cont in self.context:
            d = cont["date"] if "date" in cont else ""
            context.append(
                f"\nSource: {cont['document']}, {cont['title']}, {d}\nContext: {cont['text']}\n"
            )
        return context

    def _add_previous_context(self, prompt: str) -> str:
        """
        If there is context from a previous question, this method adds it to the prompt for the language model. It formats the context in
        the same way as the new context from the database, but separated from it and marked as prevoius context.

        :param prompt: The prompt to add the previous context to.
        :type prompt: str
        :return: The prompt with the previous context added.
        :rtype: str
        """
        if self.previous_context is not None:
            prompt += "\nContext for the answering of the previous question:\n---------------------\n"
            for cont in self.previous_context:
                prompt += self._generate_context_line(cont)
            prompt += f"\n---------------------\n"
        return prompt

    def _add_previous_chat_history(self, prompt: str) -> str:
        """
        If there is chat history from previous questions, this method adds it to the prompt for the language model to be able to answer
        questions based on the previous chat history.

        :param prompt: The prompt to add the previous chat history to.
        :type prompt: str
        :return: The prompt with the previous chat history added.
        :rtype: str
        """
        if len(self.chat_history) > 0:
            prompt += "\n\nThis is the previous chat history:\n"
            for h in self.chat_history:
                prompt += f"Question: {h['question']}"
                prompt += f"Answer: {h['answer']}"
        return prompt

    def _improve_follow_up_question(self, question1: str, question2: str) -> str:
        """
        This method employs a language model to detect whether the second question is a follow-up question to the first question. If it is,
        it enriches the second question with the context from the first such that it is understandable on its own. If it is not, it simply
        outputs the second question.

        :param question1: The first question to compare the second question to.
        :type question1: str
        :param question2: The second question to compare to the first question.
        :type question2: str
        :return: The second question, possibly enriched with context from the first question.
        :rtype: str
        """
        prompt = f"Given the following two questions, decide whether the second question is a follow-up question to the first question. "
        prompt += "If it is, enrich the second question with the context from the first such that it is understandable on its own. If it is not, simply "
        prompt += f"output the second question. ONLY OUTPUT THE QUESTION; NOTHING ELSE!\n \
        Question one: {question1}\n \
        Question two: {question2}\n \
        Question:"
        res = self.model(prompt)
        if res == question2:
            self._print(
                "The question is not a follow-up question. The model did not enrich the question."
            )
        else:
            self._print(
                f"The question is a follow-up question. The model enriched the question to: {res}"
            )
        return res

    def _detect_time_dependence(self, input: str):
        """
        This method uses a language model to detect whether the given query asks for information that was released before or after a certain
        timepoint or during a specific time period. It differentiates between the timeframe of release of the information and the timeframe of
        the information itself. It returns the time dependence in the format of a list of python dictionaries with the keys "start_date" and
        "end_date". If there is no relevant time dependence, it returns an empty list.

        :param input: The input to detect the time dependence in.
        :type input: str
        :return: The time dependence in the format of a list of python dictionaries.
        :rtype: list
        """
        # few shot prompt
        prompt = f'Detect whether the given query asks for information that was released before or after a certain timepoint or during a specific time period. Differentiate between the timeframe of release of the information and the timeframe of the information itself. We are only interested in the release timeframe. Indicate the time dependence in the format of a list of python dictionary as in the example below and only output the list, nothing else. If there is no relevant time dependence, you can return an empty list. Adhere to the format in the examples:\n\
        \n\
        Examples:\n\
        ---------------------\n\
        Question: How should the logo of the european union look like?\n\
        Answer: []\n\
        Question: What are the regulations on refrigerating appliances that were established after 1999?\n\
        Output: [{{"start_date": "01.01.2000", "end_date": None}}]\n\
        Question: What kinds of fuels were illegal in the first eight months of 2003?\n\
        Answer: [{{"start_date": "01.01.2003, "end_date": "31.08.2003"}}]\n\
        Question: Who worked on the development of e-fuels before 1992 or after 2019?\n\
        Answer: [{{"start_date": None, "end_date": "31.12.1991"}}, {{"start_date": "01.01.2019", "end_date": None}}]\n\
        ---------------------\n\
        \n\
        Question: {input}\n\
        Answer: '
        res = self.model(prompt)
        try:
            timeframes = ast.literal_eval(res)
            if len(timeframes) > 0:
                self._print(
                    f"The model detected the following timeframes: {timeframes}"
                )
            else:
                self._print("The model did not detect any timeframes.")
        except:
            try:
                splt = "[" + res.split("[")[1]
                splt = splt.split("]")[0] + "]"
                timeframes = ast.literal_eval(node_or_string=splt)
                if len(timeframes) > 0:
                    self._print(
                        f"The model detected the following timeframes: {timeframes}"
                    )
            except:
                self._print(
                    "The model did not return a valid list of python dictionaries. The output was: ",
                    res,
                )
                timeframes = []
        if not isinstance(timeframes, list):
            # in case the literal_eval succeeds but the result is not a list
            self._print(
                "The model did not return a list of python dictionaries. The output was: ",
                res,
            )
            timeframes = []
        try:
            for frame in timeframes:
                if not isinstance(frame, dict):
                    return []
                if "start_date" not in frame or "end_date" not in frame:
                    return []
                if frame["start_date"] is not None:
                    if int(frame["start_date"][6:]) > 2023:
                        return []
                if frame["end_date"] is not None:
                    if int(frame["end_date"][6:]) < 1950:
                        return []
        except:
            return []
        return timeframes

    def _extract_keywords(self, input: str):
        """
        This method uses a language model to extract important keywords from a question. Additionally, it adds one or two more keywords that
        are close to the topic of the question such that they could improve a topical keyword search. It returns the keywords as a python list
        of strings.

        :param input: The input to extract the keywords from.
        :type input: str
        :return: The extracted keywords as a python list of strings.
        :rtype: list
        """
        prompt = f"Extract the one to five most important keywords from the given query and return them as a python list of strings. Adhere to the "
        prompt += f"format in the examples.\n\
        \n\
        Examples:\n\
        ---------------------\n\
        Question: What are the regulations on refrigerating appliances that were established after 1999?\n\
        Output: ['refrigerator', 'refrigerating appliances', 'freezer']\n\
        Question: What kinds of fuels were illegal in the first eight months of 2003?\n\
        Output: ['fuels', 'illegal']\n\
        ---------------------\n\
        \n\
        Question: {input}\n\
        Output: "
        res = self.model(prompt)
        try:
            keywords = ast.literal_eval(res)
            if len(keywords) > 0:
                self._print(f"The model detected the following keywords: {keywords}")
            else:
                self._print("The model did not detect any keywords.")
        except:
            try:
                splt = "[" + res.split("[")[1]
                splt = splt.split("]")[0] + "]"
                keywords = ast.literal_eval(splt)
                if len(keywords) > 0:
                    self._print(
                        f"The model detected the following keywords: {keywords}"
                    )
            except:
                self._print(
                    "The model did not return a valid list keywords. The output was: ",
                    res,
                )
                keywords = []
        return keywords

    def clear_history(self):
        """
        This method clears the chat history of the RAG model such that it can be used to start a new conversation.
        """
        self.chat_history = []
        self.previous_context = None

    def _print(self, *args):
        """
        This method is a wrapper for the print function that only prints if the verbose attribute is set to True.
        It is used to log the processing steps of the RAG model.

        :param *args: The arguments to print. Can be any number of arguments of any type that are printable by the python print function.
        """
        if self.verbose:
            print(*args)
