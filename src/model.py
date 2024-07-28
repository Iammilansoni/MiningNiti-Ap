import os
import logging
import textwrap
from typing import Any
# from google.generativeai import configure  # Uncomment if using this module
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from langchain.chains import LLMChain  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyChatBot:
    def __init__(self, api_key: str, temperature: float):
        """
        Initializes the chatbot with the given API key and temperature setting.
        """
        os.environ['GOOGLE_API_KEY'] = api_key
        # configure(api_key=api_key)  # Uncomment if using google.generativeai
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)
        self.chain = None
        logger.info("Chatbot initialized with API key and temperature settings.")

    def set_prompt(self, template: str, input_variables: dict):
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
                raise
        else:
            logger.error("Prompt not set. Please set the prompt using set_prompt method.")
            raise Exception("Prompt not set. Please set the prompt using set_prompt method.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Allows the class instance to be called like a function.
        """
        return self.run(*args, **kwds)

# Example usage:
# bot = MyChatBot("your_api_key_here", 0.5)
# bot.set_prompt("Write a C++ code of Error handling", {})
# response = bot("Write a C++ code of Error handling")
# print(response)
