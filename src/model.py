import os
import logging
import textwrap
import yaml
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from langchain.chains import LLMChain  # type: ignore
import google.genai as genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyChatBot:
    def __init__(self, api_key: str, yaml_path: str = "conf/variables.yaml"):
        """
        Initializes the chatbot with the given API key and loads prompt template from YAML.
        """
        os.environ['GOOGLE_API_KEY'] = api_key

        # Configure Gemini client
        try:
            genai.configure(api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha"))
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)
            logger.info("Chatbot initialized with gemini-1.5-pro (v1alpha).")
        except Exception as e:
            logger.error(f"Failed to initialize gemini-1.5-pro: {str(e)}")
            logger.info("Falling back to gemini-1.5-flash.")
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

        # Load YAML config and set prompt
        try:
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)

            template = config["MiningNitiTemplate"]
            self.set_prompt(template=template, input_variables=["input"])
        except Exception as e:
            logger.error(f"Error loading YAML template: {str(e)}")
            raise

    def set_prompt(self, template: str, input_variables: list[str]):
        """
        Sets the prompt template and input variables for the chatbot.
        """
        prompt = PromptTemplate(template=template, input_variables=input_variables)
        self.chain = LLMChain(llm=self.llm, prompt=prompt)
        logger.info("Prompt template set successfully.")

    def format_code_response(self, code: str) -> str:
        """
        Formats the response cleanly.
        """
        formatted_code = textwrap.dedent(code).strip()
        return f"\n{formatted_code}\n"

    def run(self, input: str) -> Any:
        """
        Runs the chatbot with the given input.
        """
        if self.chain is not None:
            try:
                logger.info(f"Running chain with input: {input}")
                response = self.chain.predict(input=input)
                return self.format_code_response(response)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                if "404" in str(e):
                    return {"error": "Model not found. Check model name or API version."}, 400
                raise
        else:
            raise Exception("Prompt not set. Use set_prompt method first.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)
