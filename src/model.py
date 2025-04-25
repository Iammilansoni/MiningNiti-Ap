import os
import logging
import textwrap
import yaml
from typing import Any, List  # Corrected List import
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from langchain.chains import LLMChain  # type: ignore
import google.genai as genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyChatBot:
    def __init__(self, api_key: str, temperature: float, yaml_path: str = "conf/variables.yaml"):
        """
        Initializes the chatbot with the given API key, temperature,
        and loads prompt template from YAML.

        Args:
            api_key (str): Your Google API key.
            temperature (float): The temperature setting for the LLM creativity.
            yaml_path (str): Path to the YAML configuration file containing the prompt template.
        """
        os.environ['GOOGLE_API_KEY'] = api_key
        self.temperature = temperature # Store temperature if needed elsewhere, otherwise just pass it
        self.chain = None # Initialize chain attribute

        # Configure Gemini client
        try:
            # Consider removing http_options unless specifically needed and causing issues
            # genai.configure(api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha"))
            genai.configure(api_key=api_key) # Simpler configuration often works

            # Use the passed temperature parameter
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=self.temperature)
            logger.info(f"Chatbot initialized with gemini-1.5-pro and temperature={self.temperature}.")

        except Exception as e:
            logger.error(f"Failed to initialize gemini-1.5-pro: {str(e)}")
            logger.info("Falling back to gemini-1.5-flash.")
            # Use the passed temperature parameter here as well
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=self.temperature)
            logger.info(f"Chatbot initialized with gemini-1.5-flash and temperature={self.temperature}.")


        # Load YAML config and set prompt
        try:
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
                if not config or "MiningNitiTemplate" not in config:
                    raise ValueError(f"YAML file '{yaml_path}' is missing or doesn't contain 'MiningNitiTemplate'")

            template = config["MiningNitiTemplate"]
            # Assuming "input" is the only variable needed based on original code
            self.set_prompt(template=template, input_variables=["input"])
        except FileNotFoundError:
            logger.error(f"YAML configuration file not found at: {yaml_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading YAML template or setting prompt: {str(e)}")
            raise

    def set_prompt(self, template: str, input_variables: List[str]): # Use List from typing
        """
        Sets the prompt template and input variables for the chatbot.
        """
        try:
            prompt = PromptTemplate(template=template, input_variables=input_variables)
            self.chain = LLMChain(llm=self.llm, prompt=prompt)
            logger.info("Prompt template set successfully.")
        except Exception as e:
            logger.error(f"Failed to create LLMChain: {str(e)}")
            self.chain = None # Ensure chain is None if setup fails
            raise # Re-raise the exception after logging

    def format_code_response(self, code: str) -> str:
        """
        Formats the response cleanly.
        """
        # Ensure input is a string before processing
        if not isinstance(code, str):
            logger.warning(f"Received non-string response to format: {type(code)}. Converting to string.")
            code = str(code)
        formatted_code = textwrap.dedent(code).strip()
        return f"\n{formatted_code}\n"

    def run(self, input: str) -> Any:
        """
        Runs the chatbot with the given input.
        """
        if self.chain is not None:
            try:
                logger.info(f"Running chain with input: {input[:100]}...") # Log truncated input
                # Use run method for simple input/output chains if predict causes issues or for consistency
                response = self.chain.run(input=input) # Run method to ensure proper injection
                logger.info("Chain execution successful.")
                return self.format_code_response(response)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                # Specific error handling can be added here if needed
                raise Exception(f"Failed to get response from LLM: {str(e)}") # Re-raise for FastAPI handler
        else:
            logger.error("Chain not initialized. Cannot run.")
            # Raise a more specific error
            raise RuntimeError("Chatbot chain is not properly initialized. Check logs for setup errors.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # Ensure __call__ passes arguments correctly to run
        if args:
            input_arg = args[0]
            return self.run(input=input_arg)
        elif "input" in kwds:
             return self.run(input=kwds["input"])
        else:
             logger.error("No 'input' provided to chatbot call.")
             raise ValueError("Input must be provided either as a positional argument or keyword argument 'input'")
