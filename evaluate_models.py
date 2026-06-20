import numpy as np
import torch
import time
import psutil
import os
import evaluate
from tqdm import tqdm
import utils
import yaml
from transformers import set_seed

#Load Experiment Config
with open("experiment_config.yaml", "r") as file:
    experiment_config = yaml.safe_load(file)

#Set seed
seed = experiment_config["training"].get("seed", 42)
set_seed(seed)

#Set device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# === Set general input and target lengths ===
max_input_length = experiment_config["general"]["max_input_length"]
max_target_length = experiment_config["general"]["max_target_length"]

def evaluate_model(experiment):

    # Load Test and ValidationDataset
    test = utils.load_cnn_dm_split("test")
    validation = utils.load_cnn_dm_split("validation")
    model_source=experiment["model"]
    quantization_config=experiment["quantization"]

    # === Load model and tokenizer ===
    if quantization_config!='none':
        model, tokenizer, config = utils.load_model_quantized(model_source,device,quantization_config)
    else:
        model, tokenizer, config = utils.load_model(model_source,device)

    #Load decoding settings
    decoding_settings = experiment_config["decoding"]

    # Warmup the model for faster inference using validation
    for example in validation.select(range(10)):
        inputs, summary_reference = utils.preprocess(example, tokenizer,max_input_length)
        with torch.inference_mode():
            output_ids = model.generate(input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **decoding_settings)
            summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            #Print example input and generated summary to verify model is working
            print("🧾 Generated Summary:", summary)

    # === Inference and Evaluation ===
    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.synchronize()

    model.eval()
    rouge = evaluate.load("rouge")

    # === Collect summaries for ROUGE metrics ===
    generated_summaries = []
    reference_summaries = []

    # Reset memory stats after warmup
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    latencies_ms = []

    eval_dataset = test
    for example in tqdm(eval_dataset):
        inputs, summary_reference = utils.preprocess(example, tokenizer,max_input_length)

        #Initialize start and end events with timing enabled
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Synchronize CPU and GPU before timing to avoid queueing errors
            torch.cuda.synchronize()

            #Record the start event
            start_event.record()

        with torch.inference_mode():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **decoding_settings
            )

        #Record the end event
        if device.type == "cuda":
            end_event.record()
            # Synchronize again to ensure all GPU operations are complete
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)
            latencies_ms.append(elapsed_time_ms)

        # Collect generated and reference Summary details
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generated_summaries.append(generated_summary)
        reference_summaries.append(summary_reference)

    # === Efficiency Metrics ===
    total_inference_time = sum(latencies_ms) / 1000
    avg_inference_time = total_inference_time / len(eval_dataset)
    peak_memory_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
    model_size_mb = utils.get_model_size_mb(model)

    print("\n=== ⚙️ Efficiency Metrics ===")
    print(f"Total inference time: {total_inference_time:.2f} sec")
    print(f"Average time/sample: {avg_inference_time:.4f} sec")
    print(f"peak_memory_allocated_mb: {peak_memory_allocated_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    # === Save Summaries so we can view during experiment run ===
    #Clear existing file
    if os.path.exists("llm_summaries.txt"):
        os.remove("llm_summaries.txt")
    with open("llm_summaries.txt", "w") as f:
        f.write("Generated Summaries:\n")
        for s in generated_summaries:
            f.write(s + "\n\n")
        f.write("\nReference Summaries:\n")
        for s in reference_summaries:
            f.write(s + "\n\n")

    # === ROUGE ===
    rouge_results = rouge.compute(predictions=generated_summaries, references=reference_summaries)

    # Save Experiment Results
    experiment_results = {
        "model": model_source,
        "quantization": quantization_config,
        "total_inference_time": total_inference_time,
        "avg_inference_time": avg_inference_time,
        "peak_memory_allocated_mb": peak_memory_allocated_mb,
        "model_size_mb": model_size_mb,
    }

    print("\n=== ROUGE Scores ===")
    for key, value in rouge_results.items():
        print(f"{key}: {value:.4f}")
        experiment_results[key] = f"{value:.4f}"


    #Write to file
    if not os.path.exists("experiment_results"):
        os.makedirs("experiment_results")
    # delete existing file if it exists
    if os.path.exists(f"experiment_results/{experiment['id']}.yaml"):
        os.remove(f"experiment_results/{experiment['id']}.yaml")
    with open(f"experiment_results/{experiment['id']}.yaml", "w") as experiment_file:
        yaml.dump(experiment_results, experiment_file)

def main():
    experiments = experiment_config["experiments"]

    for experiment in experiments:

        if experiment['evaluate']:
            print(f"Running {experiment['id']}")
            print(f"Name: {experiment['name']}")
            print(f"Model: {experiment['model']}")
            print(f"Quantization: {experiment['quantization']}")
            evaluate_model(experiment)
            print("----------------------------------------")

if __name__ == "__main__":
    main()