import os
from abc import abstractmethod
from langchain.llms import Ollama
from openai import OpenAI
from transformers import (
    LlamaTokenizer,
    Pipeline,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)
import transformers
from torch import bfloat16
from langchain.llms import HuggingFacePipeline


class Model:
    """
    Abstract base class for all language models in the RAG-System. Provides the interface structure for the language models.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, input: str) -> str:
        pass


class OllamaModel(Model):
    """
    Wrapper for the Ollama language model, a library for running LLMs locally. For this to work, the Ollama server must be running on the port 11434 (default).
    For more information, refer to https://github.com/ollama/ollama.
    """

    model: Ollama

    def __init__(self):
        """
        Initializes the Ollama language model.
        """
        self.model = Ollama(base_url="http://localhost:11434", model="llama2:13b")

    def __call__(self, input):
        """
        Prompts the Ollama language model with the given input and returns the response.

        :param input: The input to prompt the language model with.
        :type input: str
        :return: The response from the language model.
        :rtype: str
        """
        return self.model.invoke(input)


class ChatGPT(Model):
    """
    Wrapper for the OpenAI ChatGPT language model. Requires the OPENAI_API_KEY environment variable to be set.

    :param model: The name of the language model to use, defaults to "gpt-3.5-turbo"
    :type model: str, optional
    :param system_message: The system message to be displayed at the beginning of the conversation, defaults to a generic assistant message
    :type system_message: str, optional
    """

    client: OpenAI

    def __init__(self, model="gpt-3.5-turbo", system_message=None):
        """
        Initialize the ChatGPT instance.
        """
        self.model = model
        if system_message is None:
            system_message = "You are a helpful assistant whose job it is to answer questions based on the provided information from the EURLEX database on energy law."
        self.system_message = system_message
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def __call__(self, input):
        """
        Generate a response based on the given input.

        :param input: The input to generate a response for.
        :type input: str
        :return: The generated response.
        :rtype: str
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": input},
            ],
            temperature=0.0,
        )
        response = completion.choices[0].message.content
        assert isinstance(response, str), "Invaldi response from the ChatGPT API."
        return response


class HuggingfaceLlama2(Model):
    """
    Wrapper for the Huggingface Llama2 language model.

    :param model_id: The identifier of the Llama2 model, defaults to 'meta-llama/Llama-2-13b-chat-hf'
    :type model_id: str, optional
    """

    model: AutoModelForCausalLM
    generate_text: Pipeline

    def __init__(self, model_id="meta-llama/Llama-2-13b-chat-hf"):
        """
        Initializes the Huggingface Llama2 language model.
        """
        bits_and_bytes_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_id, use_auth_token=os.getenv("HF_AUTH")
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bits_and_bytes_config,
            device_map="auto",
            token=os.getenv("HF_AUTH"),
        )
        model.eval()
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id, use_auth_token=os.getenv("HF_AUTH")
        )
        self.generate_text: Pipeline = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,
            task="text-generation",
            temperature=0.01,
            max_new_tokens=512,
            repetition_penalty=1.1,
        )
        self.llm = HuggingFacePipeline(pipeline=self.generate_text)

    def __call__(self, input: str) -> str:
        """
        Generates a response based on the given input.

        :param input: The input to generate a response for.
        :type input: str
        :return: The generated response.
        :rtype: str
        """
        return self.llm(prompt=input)
