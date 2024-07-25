import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(engine):
    tokenizer = AutoTokenizer.from_pretrained(engine)
    model = AutoModelForCausalLM.from_pretrained(engine, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def generate_explanations(utterance, in_context_demonstrations, engine, max_length=200, num_return_sequences=3):
    model, tokenizer, device = load_model_and_tokenizer(engine)
    # The prompt is adjusted to emphasize the depressive elements of both the in-context demonstrations and
    # the input utterance. This helps guide the model to focus on recognizing and explaining depressive content.
    prompt = "Below are examples with depressive elements and their explanations:\n"
    for example in in_context_demonstrations:
        prompt += f"Example: {example}\nExplanation: This example shows signs of depression because...\n"
    prompt += (f"\nAnalyze the following text for depressive elements and provide explanations:\nExample: {utterance}"
               f"\nExplanation:")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_p=0.95
        )
    explanations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return explanations


def main():
    in_context_demonstrations = [
        "She felt overwhelmed by the constant demands at work and home.",
        "He was anxious about the upcoming exams and had trouble sleeping."
    ]
    # Choose model: 'plm/Mistral-7B-Instruct-v0.2' or 'plm/gemma-7b-it'
    engine = 'plm/Mistral-7B-Instruct-v0.2'
    input_text = "I'm struggling to find motivation and everything seems pointless."

    explanations = generate_explanations(input_text, in_context_demonstrations, engine)
    for idx, explanation in enumerate(explanations, 1):
        print(f"Explanation {idx}: {explanation}")


if __name__ == '__main__':
    main()
