from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

class FinanceLLaMA:
    def __init__(self, model_path="meta-llama/Llama-2-7b"):
        """
        Initialize LLaMA model for financial applications
        Args:
            model_path: Path to the LLaMA model/weights
        """
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map="auto"  # Automatically handle device placement
        )
        
    def generate_response(self, prompt, max_length=512):
        """
        Generate response for financial analysis
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def analyze_financial_text(self, financial_text):
        """
        Analyze financial text using LLaMA
        Args:
            financial_text: Financial text to analyze
        """
        prompt = f"""Analyze the following financial text and provide insights:
        
        Text: {financial_text}
        
        Analysis:"""
        
        return self.generate_response(prompt)

llama = FinanceLLaMA()

# Example usage
financial_text = "Q4 earnings showed a 15% increase in revenue..."
analysis = llama.analyze_financial_text(financial_text)
print(analysis)
