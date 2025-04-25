import os
import logging
import textwrap
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from langchain.chains import LLMChain  # type: ignore
import google.genai as genai  # Import new SDK
from google.genai import types  # For HttpOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyChatBot:
    def __init__(self, api_key: str, temperature: float = 0.7):
        """
        Initializes the chatbot with the given API key and temperature setting.
        """
        os.environ['GOOGLE_API_KEY'] = api_key
        # Configure the new google-genai client with v1alpha API
        try:
            genai.configure(api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha"))
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=temperature)
            logger.info("Chatbot initialized with gemini-2.5-pro, v1alpha API, and temperature settings.")
        except Exception as e:
            logger.error(f"Failed to initialize model gemini-2.5-pro: {str(e)}")
            logger.info("Falling back to gemini-2.0-flash")
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
        self.chain = None

    def set_prompt(self, template: str, input_variables: list[str]):
        """
        Sets the prompt template and input variables for the chatbot.
        """
        prompt = PromptTemplate(template=template, input_variables=input_variables)
        self.chain = LLMChain(llm=self.llm, prompt=prompt)
        logger.info("Prompt template set.")

    def format_code_response(self, code: str) -> str:
        """
        Formats the code response in a structured format.
        """
        formatted_code = textwrap.dedent(code).strip()
        return f"\n{formatted_code}\n"

    def run(self, input: str) -> Any:
        """
        Runs the chatbot with the given input, if the prompt is set.
        """
        if self.chain is not None:
            try:
                response = self.chain.predict(input=input)
                logger.info("Response generated successfully.")
                logger.debug(f"Raw response: {response}")
                # Format the response before returning it
                formatted_response = self.format_code_response(response)
                logger.debug(f"Formatted response: {formatted_response}")
                return formatted_response
            except Exception as e:
                logger.error(f"Error during response generation: {str(e)}")
                if "404" in str(e):
                    return {"error": f"Model not found: {str(e)}. Check available models or API version."}, 400
                raise
        else:
            logger.error("Prompt not set. Please set the prompt using set_prompt method.")
            raise Exception("Prompt not set. Please set the prompt using set_prompt method.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Allows the class instance to be called like a function.
        """
        return self.run(*args, **kwds)
