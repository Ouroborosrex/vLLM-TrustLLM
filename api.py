from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vLLM_TrustLLM.trustllm_pkg.trustllm.generation.generation import LLMGeneration
from vLLM_TrustLLM.trustllm_pkg.trustllm import config
from vLLM_TrustLLM.trustllm_pkg.trustllm.utils.generation_utils import get_res_vllm_api

app = FastAPI()

class EvaluationRequest(BaseModel):
    model_name: str
    endpoint: str
    test_type: str = "robustness"
    data_path: str = "./API-TrustLLM/"

@app.post("/evaluate/")
def evaluate_model(request: EvaluationRequest):
    try:
        config.vllm_api_endpoint = request.endpoint

        llm_gen = LLMGeneration(
            model_path=request.model_name,
            test_type=request.test_type,
            data_path=request.data_path,
            use_vllm_api=True,
            online_model=True,
        )
        llm_gen.model_name = request.model_name
        llm_gen.generation_results()

        return {"message": "Evaluation completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")

@app.post("/generate/")
def generate_response(text: str, model_name: str, temperature: float = 0.01):
    try:
        response = get_res_vllm_api(text, model_name, temperature=temperature)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")
