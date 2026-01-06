import base64
import copy
import json
import os
import random
import time
from io import BytesIO
from mimetypes import guess_type

import cv2
import requests
from azure.identity import AzureCliCredential
from pathlib import Path
import yaml

def load_config(config_path: str):
    config_path = Path(config_path)
    if config_path.suffix in {".yaml", ".yml"}:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported config file format. Use .yaml, .yml or .json")


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Raise exceptions for any errors not specified
            except Exception as e:
                if "rate limit" in str(e).lower() or "timed out" in str(e) \
                                    or "Too Many Requests" in str(e) or "Forbidden for url" in str(e) \
                                    or "internal" in str(e).lower():
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        print("Max retries reached. Exiting.")
                        return None

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Retrying in {delay} seconds for {str(e)}...")
                    # Sleep for the delay
                    time.sleep(delay)
                else:
                    print(str(e))
                    return None

    return wrapper

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

@retry_with_exponential_backoff
def call_openai_model_with_tools(  
    messages,
    endpoints,
    model_name,
    api_key: str = None,
    use_local: bool = False, 
    tools: list = [],  # List of tool definitions
    image_paths: list = [],  
    max_tokens: int = 4096,  
    temperature: float = 0.0,  
    tool_choice: str = "auto",  # Can be "auto", "none", or a specific tool
    return_json: bool = False,
) -> dict:  
    # -------- 1. OpenAI --------
    if api_key:
        # print("[DEBUG] call_openai_model_with_tools: using OpenAI cloud")
        headers = {  
            "Content-Type": "application/json",  
            'Authorization': 'Bearer ' + api_key
        }
        endpoint = "https://api.openai.com/v1"
        url = f"{endpoint}/chat/completions"
    
    # -------- 2. Local vLLM / OpenAI compatible API --------
    elif use_local:
        # print("[DEBUG] call_openai_model_with_tools: using local vLLM")
        headers = {  
            "Content-Type": "application/json",
            # Local models usually don't need authentication, but some frameworks require a placeholder
            'Authorization': 'Bearer EMPTY'
        }
        if isinstance(endpoints, str):
            endpoint = endpoints
        elif isinstance(endpoints, list):
            endpoint = random.choice(endpoints) if endpoints else "http://localhost:8000/v1"
        else:
            raise ValueError("Endpoints must be a string or a list of strings.")
        
        # Local models use the standard OpenAI compatible path
        url = f"{endpoint}/chat/completions"
    
    # -------- 3. Azure OpenAI (fallback) --------
    else:
        # print("[DEBUG] call_openai_model_with_tools: using Azure OpenAI")
        credential = AzureCliCredential()  
        token = credential.get_token('https://cognitiveservices.azure.com/')  
        headers = {  
            "Content-Type": "application/json",  
            'Authorization': 'Bearer ' + token.token  
        }  
        if isinstance(endpoints, str):
            endpoint = endpoints
        elif isinstance(endpoints, list):
            endpoint = random.choice(endpoints)
        else:
            raise ValueError("Endpoints must be a string or a list of strings.")
        url = f"{endpoint}/openai/deployments/{model_name}/chat/completions?api-version=2025-03-01-preview"

    model = model_name

    payload = {  
        "model": model,
        "messages": copy.deepcopy(messages),  
        "temperature": temperature,
        # "max_tokens": max_tokens,
        # "reasoning_effort": reasoning_effort,
    }
    

    
    if return_json:
        payload["response_format"] = {"type": "json_object"}
  
    # Add tools to the payload if provided
    if tools:
        payload["tools"] = tools
        # payload["tool_choice"] = tool_choice
  
    if image_paths:
        # with mp.Pool(processes=min(len(image_paths), mp.cpu_count())) as pool:
        #     image_data_list = pool.map(local_image_to_data_url, image_paths)
        # image_data_list = [local_image_to_data_url(image_path) for image_path in image_paths]
        # payload['messages'].append({"role": "user", "content": []})
        # for image_data in image_data_list:
        #     payload['messages'][-1]['content'].append({"type": "image_url", "image_url": {"url": image_data}})
        payload["messages"].append({"role": "user", "content": []})
        for image_path in image_paths:
            # Convert to absolute path and add file:// prefix
            abs_path = os.path.abspath(image_path)
            file_url = f"file://{abs_path}"
            payload["messages"][-1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": file_url},
                }
            )
            # print(payload["messages"])

    response = requests.post(url, headers=headers, json=payload, timeout=600)  
  
    if response.status_code != 200:
        error_text = response.text
        raise Exception(f"OpenAI API returned status {response.status_code}: {error_text}")  
      
    response_data = response.json()  
    
    # Get the message from the response
    message = response_data['choices'][0]['message']
    
    # Check if there's a tool call in the response
    if "tool_calls" in message:
        # Return the entire message object when tools are being used
        return message
    else:
        # If there's no tool call, just return the text content
        return {"content": message['content'].strip(), "tool_calls": None}

class AzureOpenAIEmbeddingService:  
    @staticmethod  
    @retry_with_exponential_backoff
    def get_embeddings(endpoints, model_name, input_text, api_key: str = None, use_local=False):  
        """  
        Call Azure OpenAI Embedding service and get embeddings for the input text.  
  
        :param api_key: Your Azure OpenAI API key.  
        :param endpoint: The endpoint URL for the OpenAI service.  
        :param model: The model name for generating embeddings.  
        :param input_text: The text for which you want to generate embeddings.  
        :return: The embeddings as a JSON response.  
        """  
        if api_key:
            headers = {  
                "Content-Type": "application/json",  
                'Authorization': 'Bearer ' + api_key
            }
            endpoint = "https://api.openai.com/v1"
            url = f"{endpoint}/embeddings"
        elif use_local:
            headers = {  
                "Content-Type": "application/json",
                
                'Authorization': 'Bearer EMPTY'
            }
            if isinstance(endpoints, str):
                endpoint = endpoints
            elif isinstance(endpoints, list):
                endpoint = random.choice(endpoints)
            else:
                raise ValueError("Endpoints must be a string or a list of strings.")
            
            url = f"{endpoint}/embeddings"
        else:
            if isinstance(endpoints, str):
                endpoint = endpoints
            elif isinstance(endpoints, list):
                endpoint = random.choice(endpoints)
            else:
                raise ValueError("Endpoints must be a string or a list of strings.")  
            # Define the URL for the embeddings endpoint  
            url = f"{endpoint}/openai/deployments/{model_name}/embeddings?api-version=2023-05-15"  
    
            credential = AzureCliCredential()  
            token = credential.get_token('https://cognitiveservices.azure.com/')  
            headers = {  
                "Content-Type": "application/json",  
                'Authorization': 'Bearer ' + token.token  
            }
        
        model = model_name
        # Set up the payload for the request  
        payload = {  
            "input": input_text,
            "model": model
        }  
  
        # Make the request to the Azure OpenAI service  
        response = requests.post(url, headers=headers, data=json.dumps(payload))  
  
        # Check if the request was successful  
        if response.status_code == 200:  
            return response.json()['data']
        else:  
            response.raise_for_status()

def extract_answer(message: dict) -> str | None:
    """
    Extract the plain-text answer from an assistant message that may include
    tool calls.

    The function first checks the normal `content` field (for responses that
    are not using tools). If the assistant responded via a tool call, it
    attempts to parse the JSON string stored in
    `message["tool_calls"][i]["function"]["arguments"]` and returns the value
    associated with the key `"answer"`.

    Parameters
    ----------
    message : dict
        The assistant message returned by `call_openai_model_with_tools`.

    Returns
    -------
    str | None
        The extracted answer, or ``None`` if no answer could be found.
    """
    # Tool-based response
    for call in message.get("tool_calls", []):
        args_json = call["function"]["arguments"]
        args = json.loads(args_json)
        if (answer := args.get("answer")):
            return answer

    # Direct text response
    if (content := message.get("content")):
        return content.strip()
    
    return None


if __name__ == "__main__":
    # Example for Azure
    # call_openai_model_with_tools(
    #     messages=[{"role": "user", "content": "Hello, how are you?"}],
    #     endpoints=["https://msra-im-openai-eus2.openai.azure.com"],
    #     model_name="o3",
    #     tools=[],
    #     image_paths=[],
    #     max_tokens=4096,
    #     temperature=0.0,
    #     tool_choice="auto",
    #     return_json=False,
    # )

    # Example for OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        response = call_openai_model_with_tools(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            endpoints=None, # Not used for OpenAI
            model_name="gpt-4o",
            api_key=api_key,
            tools=[],
            image_paths=[],
            max_tokens=4096,
            temperature=0.0,
            tool_choice="auto",
            return_json=False,
        )
        print(response)
    else:
        print("OPENAI_API_KEY environment variable not set.")