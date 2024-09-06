from openai import OpenAI, AzureOpenAI
from transformers import AutoTokenizer, AutoModel
import torch
import os
import logging
from tqdm import tqdm
import trustllm_pkg.trustllm.config
from trustllm_pkg.trustllm.utils import file_process
from tenacity import retry, wait_random_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


class DataEmbedder:
    """
    A class for embedding textual data using OpenAI's or HuggingFace's embedding models.
    """

    def __init__(self, model=None, save_dir='saved_embeddings'):
        """
        Initialize the DataEmbedder class.

        Args:
            save_dir (str): Directory to save the embedding results.
        """
        self.save_dir = save_dir
        # Create the directory if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # HuggingFace model and tokenizer initialization (if used)
        # self.hf_tokenizer = None
        # self.hf_model = None
        self.use_hf_embedder = False
        if model is not None:
            hf_model_name = model
            self.use_hf_embedder = True
            self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            self.hf_model = AutoModel.from_pretrained(hf_model_name)
            print(f'Model {self.hf_model} loaded!')

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
    def get_embeddings(self, string, embedding_model='text-embedding-ada-002'):
        """
        Retrieve embeddings from either OpenAI or HuggingFace model.

        Args:
            string (str): Input string to embed.
            embedding_model (str): OpenAI embedding model name (optional if using HuggingFace).
        
        Returns:
            Embedding vector.
        """
        # Use HuggingFace embedder if configured
        if self.use_hf_embedder:
            logging.info("Using HuggingFace model for embedding.")
            return self.get_hf_embeddings(string)
        
        # Use Azure OpenAI or OpenAI embeddings
        if trustllm_pkg.trustllm.config.azure_openai:
            azure_endpoint = trustllm_pkg.trustllm.config.azure_api_base
            api_key = trustllm_pkg.trustllm.config.azure_api_key
            api_version = trustllm_pkg.trustllm.config.azure_api_version
            model = trustllm_pkg.trustllm.config.azure_embedding_engine
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            response = client.embeddings.create(
                model=model,
                input=string
            )
        else:
            api_key = trustllm_pkg.trustllm.config.openai_key
            if trustllm_pkg.trustllm.config.openai_api_base is not None:
                client = OpenAI(api_key=api_key, base_url=trustllm_pkg.trustllm.config.openai_api_base)
            else:
                client = OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model=embedding_model,
                input=string
            )

        return response.data[0].embedding

    def get_hf_embeddings(self, string):
        """
        Get embeddings using a HuggingFace model.

        Args:
            string (str): Input string to embed.

        Returns:
            list: Embedding vector as a list.
        """
        inputs = self.hf_tokenizer(string, return_tensors='pt')
        with torch.no_grad():
            model_output = self.hf_model(**inputs)
        embeddings = model_output.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings

    def save_embeddings(self, embeddings, filename):
        """
        Save embeddings to a JSON file.

        Args:
            embeddings: The embeddings to be saved.
            filename (str): The filename for saving the embeddings.
        """
        save_path = os.path.join(self.save_dir, filename)
        file_process.save_json(embeddings, save_path)
        logging.info("Embeddings saved to %s", save_path)
        print("Embeddings saved to %s", save_path)

    def embed_data(self, data, filename='embeddings.json', resume=False):
        """
        Embed a dataset and save the embeddings.

        Args:
            data: List of data to be embedded.
            filename (str): The filename for saving embeddings. Default is 'embeddings.json'.
            resume (bool): Flag to resume from saved progress. Default is False.

        Returns:
            str: Path to the saved embeddings file.
        """
        assert isinstance(data, list), "Data must be a list."
        logging.info('Evaluating...')
        if resume:
            try:
                data = file_process.load_json(os.path.join(self.save_dir, filename))
                logging.info("Resuming from saved progress.")
            except FileNotFoundError:
                logging.warning("No saved progress file found. Starting from scratch.")

        try:
            embedded_data = self.parallel_embedding(data, self.get_embeddings, filename)
            self.save_embeddings(embedded_data, filename)
        except Exception as error:
            logging.error("Failed processing with error: %s", str(error))

        return os.path.join(self.save_dir, filename)

    def parallel_embedding(self, data, embedding_func, filename):
        with ThreadPoolExecutor(max_workers=trustllm_pkg.trustllm.config.max_worker_embedding) as executor:
            future_to_data = {executor.submit(self.embed_text, el, embedding_func): el for el in data}
            results = []
            for future in tqdm(as_completed(future_to_data), total=len(data)):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logging.error("An error occurred: %s", str(e))
                    self.save_embeddings(data, filename)
                    raise

        return results

    def embed_text(self, data_element, embedding_func):
        try:
            if 'embedding' not in data_element:
                data_element['embedding'] = embedding_func(data_element['res'])
                logging.info("Processed text: %s", data_element.get('res', ''))
            return data_element
        except Exception as e:
            logging.error("Error embedding text %s: %s", data_element.get('res', ''), str(e))
            raise
