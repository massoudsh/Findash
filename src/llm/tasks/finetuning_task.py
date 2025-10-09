from src.core.celery_app import celery_app
from .. import training

@celery_app.task(name="llm.trigger_finetuning")
def trigger_finetuning(output_dir: str = "./peft-output"):
    """
    Celery task to start a fine-tuning job for the language model.

    Args:
        output_dir (str): The directory where the fine-tuned model
                          and tokenizer will be saved.
    """
    try:
        print(f"Starting fine-tuning job. Output will be saved to {output_dir}")
        training.run_training(output_dir=output_dir)
        print("Fine-tuning job finished successfully.")
        return {"status": "success", "output_dir": output_dir}
    except Exception as e:
        print(f"Error during fine-tuning job: {e}")
        return {"status": "error", "error_message": str(e)} 