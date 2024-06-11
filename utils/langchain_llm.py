import base64
import os
import yaml
import requests
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from utils.gcp import get_creds
from langchain_google_vertexai import ChatVertexAI


# TODO temperature and other parameters
class LangchainCiscoGPT4:
    """
    This class is used to build Cisco GPT-4 model for langchain framework.
    methods:
        cisco_gpt4(): returns the AzureChatOpenAI object for langchain framework
    """
    with open('secrets/secrets.yaml', 'r') as file:
        config = yaml.safe_load(file)

    OPENAI_URL = config['cisco_gpt4']['url']
    OPENAI_PAYLOAD = config['cisco_gpt4']['payload']
    OPENAI_AZURE_ENDPOINT = config['cisco_gpt4']['azure_endpoint']
    OPENAI_API_VERSION = config['cisco_gpt4']['api_version']
    OPENAI_MODEL_VERSION = config['cisco_gpt4']['model']
    OPENAI_SEED = config['cisco_gpt4']['seed']

    def __init__(self):
        self.creds = get_creds()

    def get_app_key(self):
        return self.creds['app_key']

    def get_api_key(self):
        CLIENT_ID = self.creds['client_id']
        CLIENT_SECRET = self.creds['client_secret']

        value = base64.b64encode(
            f'{CLIENT_ID}:{CLIENT_SECRET}'.encode('utf-8')).decode('utf-8')
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}"
        }

        response = requests.post(
            self.OPENAI_URL, headers=headers, data=self.OPENAI_PAYLOAD)

        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            response.raise_for_status()

    def cisco_gpt4(self):
        return AzureChatOpenAI(
            azure_deployment=self.OPENAI_MODEL_VERSION,
            azure_endpoint=self.OPENAI_AZURE_ENDPOINT,
            api_key=self.get_api_key(),
            openai_api_version=self.OPENAI_API_VERSION,
            model_kwargs={
                "user": f'{{"appkey": "{self.get_app_key()}"}}'
            }
        )


class LangchainOpenAI:
    """
    This class is used to build OpenAI model for langchain framework.
    methods:
        openai(): returns the OpenAI object for langchain framework
    """
    with open('secrets/secrets.yaml', 'r') as file:
        config = yaml.safe_load(file)

    os.environ["OPENAI_ORGANIZATION"] = config['open_ai']['organization']
    os.environ["OPENAI_PROJECT_ID"] = config['open_ai']['project_id']
    os.environ["OPENAI_API_KEY"] = config['open_ai']['api_key']

    def openai(self, model_name="gpt-3.5-turbo"):
        """
        model_name could be gpt-4o
        """
        return ChatOpenAI(model=model_name,
                          temperature=0,
                          max_tokens=256,
                          timeout=None,
                          max_retries=2)


class LangchainGermini:
    def germini(self):
        return ChatVertexAI(model="gemini-pro")
