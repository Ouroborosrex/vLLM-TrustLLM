from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from trustllm_pkg.trustllm.generation.generation import LLMGeneration
from trustllm_pkg.trustllm import config
from trustllm_pkg.trustllm.utils.generation_utils import get_res_vllm_api
from trustllm_pkg.trustllm.task import robustness, privacy, ethics, fairness, truthfulness, safety
from trustllm_pkg.trustllm.utils import file_process
import os
import json

app = FastAPI()

# Use uvicorn api:app --host 0.0.0.0 --port 8080 --reload 

class EvaluationRequest(BaseModel):
    model_name: str
    endpoint: str
    test_type: str = "robustness"
    data_path: str = "./dataset/"

class ModelNameRequest(BaseModel):
    model_name: str
    embedder_model: str = 'allenai/longformer-base-4096'



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

@app.post("/evaluate/robustness")
def evaluate_robustness(request: ModelNameRequest):
    try:
        result_file_path = os.path.join(f'generation_results/{request.model_name}/robustness/', "robustness_evaluation_results.json")
        
        # Check if results already exist
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as result_file:
                existing_results = json.load(result_file)
            print("Returning existing robustness evaluation results.")
            return existing_results


        evaluator = robustness.RobustnessEval()
        advglue_data = file_process.load_json(f"generation_results/{request.model_name}/robustness/AdvGLUE.json")

        advinstruction_data = file_process.load_json(f"generation_results/{request.model_name}/robustness/AdvInstruction.json")
        ood_detection_data = file_process.load_json(f"generation_results/{request.model_name}/robustness/ood_detection.json")
        ood_generalization_data = file_process.load_json(f"generation_results/{request.model_name}/robustness/ood_generalization.json")

        try:
            advglue_result = evaluator.advglue_eval(advglue_data)
        except:
            advglue_result = 'N/A'
        try:
            advinstruction_result = evaluator.advinstruction_eval(advinstruction_data, custom_model=request.embedder_model)
        except:
            advinstruction_result = 'N/A'
        try:
            ood_detection_result = evaluator.ood_detection(ood_detection_data)
        except Exception as e:
            print('ood_detection'+e)
            ood_detection_result = 'N/A'
        try:
            ood_generalization_result = evaluator.ood_generalization(ood_generalization_data)
        except Exception as e:
            print('ood_generalization'+e)
            ood_generalization_result = 'N/A'
        
        results = {
            "advglue_result": advglue_result,
            "advinstruction_result": advinstruction_result,
            "ood_detection_result": ood_detection_result,
            "ood_generalization_result": ood_generalization_result
        }
        result_file_path = os.path.join(f'generation_results/{request.model_name}/robustness/', "robustness_evaluation_results.json")
        with open(result_file_path, 'w') as result_file:
            json.dump(results, result_file, indent=4)
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error during robustness evaluation: {str(e)}")

@app.post("/evaluate/privacy")
def evaluate_privacy(request: ModelNameRequest):
    try:
        result_file_path = os.path.join(f'generation_results/{request.model_name}/privacy/', "privacy_evaluation_results.json")
        
        # Check if results already exist
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as result_file:
                existing_results = json.load(result_file)
            print("Returning existing privacy evaluation results.")
            return existing_results

        evaluator = privacy.PrivacyEval()
        privacy_confAIde_data = file_process.load_json(f"generation_results/{request.model_name}/privacy/privacy_confAIde_data.json")
        privacy_awareness_query_data = file_process.load_json(f"generation_results/{request.model_name}/privacy/privacy_awareness_query_data.json")
        privacy_leakage_data = file_process.load_json(f"generation_results/{request.model_name}/privacy/privacy_leakage_data.json")

        try: 
            confAIDe_result = evaluator.ConfAIDe_eval(privacy_confAIde_data)
        except:
            confAIDe_result = 'N/A'
        try:
            awareness_query_normal_result = evaluator.awareness_query_eval(privacy_awareness_query_data, type='normal')
        except:
            awareness_query_normal_result = 'N/A'
        try:
            awareness_query_aug_result = evaluator.awareness_query_eval(privacy_awareness_query_data, type='aug')
        except:
            awareness_query_aug_result = 'N/A'
        try:
            leakage_result = evaluator.leakage_eval(privacy_leakage_data)
        except:
            leakage_result = 'N/A'

        # Save results to files
        results = {
            "confAIDe_result": confAIDe_result,
            "awareness_query_normal_result": awareness_query_normal_result,
            "awareness_query_aug_result": awareness_query_aug_result,
            "leakage_result": leakage_result
        }

        result_file_path = os.path.join(f'generation_results/{request.model_name}/privacy/', "privacy_evaluation_results.json")
        with open(result_file_path, 'w') as result_file:
            json.dump(results, result_file, indent=4)

        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during privacy evaluation: {str(e)}")

@app.post("/evaluate/ethics")
def evaluate_ethics(request: ModelNameRequest):
    try:
        result_file_path = os.path.join(f'generation_results/{request.model_name}/ethics/', "ethics_evaluation_results.json")
        
        # Check if results already exist
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as result_file:
                existing_results = json.load(result_file)
            print("Returning existing ethics evaluation results.")
            return existing_results
        
        evaluator = ethics.EthicsEval()
        explicit_ethics_data = file_process.load_json(f"generation_results/{request.model_name}/ethics/explicit_ethics_data.json")
        implicit_ethics_data = file_process.load_json(f"generation_results/{request.model_name}/ethics/implicit_ethics_data.json")
        awareness_data = file_process.load_json(f"generation_results/{request.model_name}/ethics/awareness_data.json")

        try:
            explicit_ethics_low_result = evaluator.explicit_ethics_eval(explicit_ethics_data, eval_type='low')
        except:
            explicit_ethics_low_result = 'N/A'
        try:
            explicit_ethics_high_result = evaluator.explicit_ethics_eval(explicit_ethics_data, eval_type='high')
        except:
            explicit_ethics_high_result = 'N/A'
        try:
            implicit_ethics_ETHICS_result = evaluator.implicit_ethics_eval(implicit_ethics_data, eval_type='ETHICS')
        except:
            implicit_ethics_ETHICS_result = 'N/A'
        try:
            implicit_ethics_social_norm_result = evaluator.implicit_ethics_eval(implicit_ethics_data, eval_type='social_norm')
        except:
            implicit_ethics_social_norm_result = 'N/A'
        try:
            awareness_result = evaluator.awareness_eval(awareness_data)
        except:
            awareness_result = 'N/A'

        results =  {
            "explicit_ethics_low_result": explicit_ethics_low_result,
            "explicit_ethics_high_result": explicit_ethics_high_result,
            "implicit_ethics_ETHICS_result": implicit_ethics_ETHICS_result,
            "implicit_ethics_social_norm_result": implicit_ethics_social_norm_result,
            "awareness_result": awareness_result
        }
    
        result_file_path = os.path.join(f'generation_results/{request.model_name}/ethics/', "ethics_evaluation_results.json")
        with open(result_file_path, 'w') as result_file:
            json.dump(results, result_file, indent=4)

        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during ethics evaluation: {str(e)}")

@app.post("/evaluate/fairness")
def evaluate_fairness(request: ModelNameRequest):
    try:
        result_file_path = os.path.join(f'generation_results/{request.model_name}/fairness/', "fairness_evaluation_results.json")
        
        # Check if results already exist
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as result_file:
                existing_results = json.load(result_file)
            print("Returning existing fairness evaluation results.")
            return existing_results
        
        evaluator = fairness.FairnessEval()
        stereotype_recognition_data = file_process.load_json(f"generation_results/{request.model_name}/fairness/stereotype_recognition_data.json")
        stereotype_agreement_data = file_process.load_json(f"generation_results/{request.model_name}/fairness/stereotype_agreement_data.json")
        stereotype_query_test_data = file_process.load_json(f"generation_results/{request.model_name}/fairness/stereotype_query_test_data.json")
        disparagement_data = file_process.load_json(f"generation_results/{request.model_name}/fairness/disparagement_data.json")
        preference_data = file_process.load_json(f"generation_results/{request.model_name}/fairness/preference_data.json")

        try:
            stereotype_recognition_result = evaluator.stereotype_recognition_eval(stereotype_recognition_data)
        except:
            stereotype_recognition_result = 'N/A'
        try:
            stereotype_agreement_result = evaluator.stereotype_agreement_eval(stereotype_agreement_data)
        except:
            stereotype_agreement_result = 'N/A'
        try:
            stereotype_query_result = evaluator.stereotype_query_eval(stereotype_query_test_data)
        except:
            stereotype_query_result = 'N/A'
        try:
            disparagement_result = evaluator.disparagement_eval(disparagement_data)
        except:
            disparagement_result = 'N/A'
        try:
            preference_result = evaluator.preference_eval(preference_data)
        except:
            preference_result = 'N/A'

        results = {
            "stereotype_recognition_result": stereotype_recognition_result,
            "stereotype_agreement_result": stereotype_agreement_result,
            "stereotype_query_result": stereotype_query_result,
            "disparagement_result": disparagement_result,
            "preference_result": preference_result
        }

        result_file_path = os.path.join(f'generation_results/{request.model_name}/fairness/', "fairness_evaluation_results.json")
        with open(result_file_path, 'w') as result_file:
            json.dump(results, result_file, indent=4)

        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during fairness evaluation: {str(e)}")


@app.post("/evaluate/truthfulness")
def evaluate_truthfulness(request: ModelNameRequest):
    try:
        result_file_path = os.path.join(f'generation_results/{request.model_name}/truthfulness/', "truthfulness_evaluation_results.json")
        
        # Check if results already exist
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as result_file:
                existing_results = json.load(result_file)
            print("Returning existing truthfulness evaluation results.")
            return existing_results
        
        evaluator = truthfulness.TruthfulnessEval()
        misinformation_internal_data = file_process.load_json(f"generation_results/{request.model_name}/truthfulness/misinformation_internal_data.json")
        misinformation_external_data = file_process.load_json(f"generation_results/{request.model_name}/truthfulness/misinformation_external_data.json")
        hallucination_data = file_process.load_json(f"generation_results/{request.model_name}/truthfulness/hallucination_data.json")
        sycophancy_data = file_process.load_json(f"generation_results/{request.model_name}/truthfulness/sycophancy_data.json")
        adv_fact_data = file_process.load_json(f"generation_results/{request.model_name}/truthfulness/adv_fact_data.json")

        try:
            misinformation_internal_result = evaluator.internal_eval(misinformation_internal_data)
        except:
            misinformation_internal_result = 'N/A'
        try:
            misinformation_external_result = evaluator.external_eval(misinformation_external_data)
        except:
            misinformation_external_result = 'N/A'
        try:
            hallucination_result = evaluator.hallucination_eval(hallucination_data)
        except:
            hallucination_result = 'N/A'
        try:
            sycophancy_persona_result = evaluator.sycophancy_eval(sycophancy_data, eval_type='persona')
        except:
            sycophancy_persona_result = 'N/A'
        try:
            sycophancy_preference_result = evaluator.sycophancy_eval(sycophancy_data, eval_type='preference')
        except:
            sycophancy_preference_result = 'N/A'
        try:
            advfact_result = evaluator.advfact_eval(adv_fact_data)
        except:
            advfact_result = 'N/A'

        results =  {
            "misinformation_internal_result": misinformation_internal_result,
            "misinformation_external_result": misinformation_external_result,
            "hallucination_result": hallucination_result,
            "sycophancy_persona_result": sycophancy_persona_result,
            "sycophancy_preference_result": sycophancy_preference_result,
            "advfact_result": advfact_result
        }

        result_file_path = os.path.join(f'generation_results/{request.model_name}/truthfulness/', "truthfulness_evaluation_results.json")
        with open(result_file_path, 'w') as result_file:
            json.dump(results, result_file, indent=4)

        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during truthfulness evaluation: {str(e)}")


@app.post("/evaluate/safety")
def evaluate_safety(request: ModelNameRequest):
    try:
        result_file_path = os.path.join(f'generation_results/{request.model_name}/safety/', "safety_evaluation_results.json")
        
        # Check if results already exist
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as result_file:
                existing_results = json.load(result_file)
            print("Returning existing safety evaluation results.")
            return existing_results
        
        evaluator = safety.SafetyEval()
        jailbreak_data = file_process.load_json(f"generation_results/{request.model_name}/safety/jailbreak_data.json")
        exaggerated_data = file_process.load_json(f"generation_results/{request.model_name}/safety/exaggerated_data.json")
        misuse_data = file_process.load_json(f"generation_results/{request.model_name}/safety/misuse_data.json")

        try:
            jailbreak_total_result = evaluator.jailbreak_eval(jailbreak_data, eval_type='total')
        except:
            jailbreak_total_result = 'N/A'
        try:
            jailbreak_single_result = evaluator.jailbreak_eval(jailbreak_data, eval_type='single')
        except:
            jailbreak_single_result = 'N/A'
        try:
            exaggerated_result = evaluator.exaggerated_eval(exaggerated_data)
        except:
            exaggerated_result = 'N/A'
        try:
            misuse_result = evaluator.misuse_eval(misuse_data)
        except:
            misuse_result = 'N/A'

        results = {
            "jailbreak_total_result": jailbreak_total_result,
            "jailbreak_single_result": jailbreak_single_result,
            "exaggerated_result": exaggerated_result,
            "misuse_result": misuse_result
        }

        result_file_path = os.path.join(f'generation_results/{request.model_name}/safety/', "safety_evaluation_results.json")
        with open(result_file_path, 'w') as result_file:
            json.dump(results, result_file, indent=4)

        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during safety evaluation: {str(e)}")
