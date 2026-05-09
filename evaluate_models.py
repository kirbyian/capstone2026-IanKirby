import torch
import time
import psutil
import os
import evaluate
from tqdm import tqdm
import utils
import yaml


#Load Experiment Config
with open("experiment_config.yaml", "r") as file:
    experiment_config = yaml.safe_load(file)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# === Preprocess ===
max_input_length = experiment_config["general"]["max_input_length"]
max_target_length = experiment_config["general"]["max_target_length"]

def evaluate_model(model_source, quantization_config):

    # Get Datasets
    test = utils.load_cnn_dm_split("test")

    # === Load model and tokenizer ===
    if quantization_config!='none':
        model, tokenizer, config = utils.load_model_quantized(model_source,device,quantization_config)
    else:
        model, tokenizer, config = utils.load_model(model_source,device)

    model.eval()
    # === Simple Example Summary to quickly test model ===
    input_text = "The Eiffel Tower is located in Paris and is one of the most famous landmarks in the world."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=64)
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("📝 Example Input:", input_text)
    print("🧾 Generated Summary:", summary)

    rouge = evaluate.load("rouge")

    # === Inference and Evaluation ===
    generated_summaries = []
    reference_summaries = []

    start_time = time.time()
    mem_before = psutil.Process(os.getpid()).memory_info().rss / 1e6  # MB

    for example in tqdm(test.select(range(20))):
        inputs, reference = utils.preprocess(example, tokenizer,max_input_length)
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generated_summaries.append(summary)
        reference_summaries.append(reference)

    end_time = time.time()
    mem_after = psutil.Process(os.getpid()).memory_info().rss / 1e6  # MB

    # === Save Summaries ===
    with open("llm_summaries.txt", "w") as f:
        f.write("Generated Summaries:\n")
        for s in generated_summaries:
            f.write(s + "\n\n")
        f.write("\nReference Summaries:\n")
        for s in reference_summaries:
            f.write(s + "\n\n")

    # === ROUGE ===
    rouge_results = rouge.compute(predictions=generated_summaries, references=reference_summaries)

    print("\n=== 📊 ROUGE Scores ===")
    for key, scores in rouge_results.items():
        if isinstance(scores, dict):  # detailed metrics
            print(
                f"{key}: Precision={scores['precision']:.4f}, Recall={scores['recall']:.4f}, F1={scores['fmeasure']:.4f}")

    # === Efficiency Metrics ===
    inference_time = end_time - start_time
    avg_time = inference_time / len(test)
    memory_used = mem_after - mem_before
    model_size_mb = utils.get_model_size_mb(model)

    print("\n=== ⚙️ Efficiency Metrics ===")
    print(f"Total inference time: {inference_time:.2f} sec")
    print(f"Average time/sample: {avg_time:.4f} sec")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")


def main():
    experiments = experiment_config["experiments"]

    for experiment in experiments:

        if experiment['evaluate']:
            print(f"Running {experiment['id']}")
            print(f"Name: {experiment['name']}")
            print(f"Model: {experiment['model']}")
            print(f"Quantization: {experiment['quantization']}")
            evaluate_model(experiment["model"], experiment["quantization"])

if __name__ == "__main__":
    main()