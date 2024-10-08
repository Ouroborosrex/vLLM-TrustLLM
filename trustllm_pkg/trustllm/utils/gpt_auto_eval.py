from openai import OpenAI,AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from trustllm_pkg.trustllm.utils import file_process
import logging
import os
import trustllm_pkg.trustllm
import concurrent.futures
import trustllm_pkg.trustllm.config

# Setting up basic logging configuration
logging.basicConfig(filename='autoevaluator.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


#Retry decorator with exponential backoff and stop condition for API calls
# @retry(wait=wait_random_exponential(min=5, max=20))
def get_res(string, model='gpt-4', temperature=0, message=None):
    """
    Retrieve a response from the selected evaluation model.

    Args:
        string (str): The input string to process.
        model (str): The model to use for generating the response.
        temp (float): The temperature setting for the API request.

    Returns:
        str: The API response content.

    Raises:
        ValueError: If the API response is null or an empty string.
    """
    try:
        if message is None:
            message = [{"role": "user", "content": string}]
        
        # Check if vLLM is selected
        if model != "gpt-4":
            print('model', model)
            api_key = trustllm_pkg.trustllm.config.vllm_key
            base_url = trustllm_pkg.trustllm.config.vllm_api_endpoint
            print(base_url, api_key)
            client = OpenAI(api_key=api_key, base_url=base_url)
            print(message)
            response = client.chat.completions.create(model=model, messages=message, temperature=temperature, max_tokens=15)
            print(response)
            res = response.choices[0].message.content if response.choices[0].message.content else ValueError("Empty response from API")
            print(response)
            return res
        
        elif trustllm_pkg.trustllm.config.azure_openai:
            azure_endpoint = trustllm_pkg.trustllm.config.azure_api_base
            api_key = trustllm_pkg.trustllm.config.azure_api_key
            api_version = trustllm_pkg.trustllm.config.azure_api_version
            model = trustllm_pkg.trustllm.config.azure_engine
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            stream = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature
            )
        else:
            api_key = trustllm_pkg.trustllm.config.openai_key
            if trustllm_pkg.trustllm.config.openai_api_base is not None:
                client = OpenAI(
                    api_key=api_key,
                    base_url=trustllm_pkg.trustllm.config.openai_api_base
                )
            else:
                client = OpenAI(api_key=api_key)
            stream = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature,
            )
        
        # PROBLEM IS HERE #
        if not stream.choices[0].message.content:
            raise ValueError("The response from the API is NULL or an empty string!")
        response = stream.choices[0].message.content

    except Exception as e:
        print(e)
        return None
    return response


class AutoEvaluator:
    """
    A class for automating the evaluation of text using the OpenAI API.
    """

    def __init__(self, save_dir='saved_evaluations'):
        """
        Initialize the AutoEvaluator class.

        Args:
            save_dir (str): Directory for saving evaluation results.
        """
        self.save_dir = save_dir
        self.max_worker = trustllm_pkg.trustllm.config.max_worker_auto_eval
        self.evaluation_llm = trustllm_pkg.trustllm.config.evaluation_llm
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        #openai.api_key = trustllm.config.openai_key

    def save_progress(self, data, filename='auto_eval.json'):
        """
        Save evaluation progress to a JSON file.

        Args:
            data: Data to be saved.
            filename (str): Name of the file for saving the data.
        """
        save_path = os.path.join(self.save_dir, filename)
        file_process.save_json(data, save_path)
        logging.info("Progress saved to %s", save_path)

    def evaluate(self, data, task, resume=False, progress_filename='eval_progress.json', concat=True):
        """
        Evaluate a given dataset using a specified task.

        Args:
            data: Data to be evaluated.
            task (str): The task identifier for the evaluation.
            resume (bool): Flag to resume from saved progress. Default is False.
            progress_filename (str): The filename for saving or resuming progress.
            concat (bool): Flag to concatenate responses. Default is True.

        Returns:
            The evaluated data.
        """

        def save_progress_callback(future):
            if future.exception() is not None:
                logging.error("An error occurred: %s", str(future.exception()))
                # Save progress in case of an error
                self.save_progress(data, filename=progress_filename)

        def process_item(item, el):
            try:
                if 'eval_res' not in el:

                    # print('Prompt: {}'.format(item))
                    eval_res = get_res(item, model=self.evaluate)
                    print('Response: {}'.format(eval_res))
                    el['eval_res'] = eval_res
                    logging.info("Evaluated item: %s", item)
                    logging.info("Evaluated result: %s", eval_res)
            except Exception as e:
                logging.error("Error processing item %s: %s", item, str(e))
                # self.save_progress(data, filename=progress_filename)
                raise

        task_prompt_dict = trustllm_pkg.trustllm.config.task_prompt
        prompt_data = []

        if not concat:
            replace_dict = task_prompt_dict.get(task, {}).get('mapping', {})
            prompt = task_prompt_dict.get(task, {}).get('prompt', '')
            for el in data:
                single_prompt = prompt
                for k, v in replace_dict.items():
                    single_prompt = single_prompt.replace(k, str(el[v]))
                prompt_data.append(single_prompt)
        else:
            prompt = task_prompt_dict.get(task, {}).get('prompt', '')
            prompt_data = [prompt + item['res'] for item in data]

        if resume:
            load_path = os.path.join(self.save_dir, progress_filename)
            try:
                data = file_process.load_json(load_path)
                logging.info("Resuming evaluation from saved progress.")
            except FileNotFoundError:
                logging.warning("No saved progress file found at %s. Starting a new evaluation.", load_path)

        assert isinstance(data, list), "Data must be a list."
        assert task is not None, "Task must be specified for evaluation."

        logging.info('Total data number: %d', len(data))
        logging.info('Evaluating...')

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            futures = [executor.submit(process_item, item, el) for item, el in zip(prompt_data, data)]

            # Add a callback to handle completion and errors
            for future in concurrent.futures.as_completed(futures):
                future.add_done_callback(save_progress_callback)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

        self.save_progress(data, filename=progress_filename)
        return data
