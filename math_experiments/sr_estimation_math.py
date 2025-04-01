# Standard library imports
import argparse
import os
import pickle
from collections import deque

# Third-party imports
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --------------------------
# Model Utility Functions
# --------------------------

def print_trainable_parameters(model_parameters, name):
    """
    Prints the number of trainable parameters in the model.
    
    Args:
        model_parameters: Iterator over model parameters with names
        name: Name to display in the output
    """
    trainable_params = 0
    all_param = 0
    for n, param in model_parameters:
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"({name}) trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

def get_filtered_params(model, name_filter, return_with_names=False):
    """
    Get parameters filtered by name.
    
    Args:
        model: The model to filter parameters from
        name_filter: Function to filter parameter names
        return_with_names: Whether to return names with parameters
        
    Returns:
        Filtered parameters (with names if requested)
    """
    if return_with_names:
        return filter(lambda p: name_filter(p[0]), model.named_parameters())
    else:
        return [p for n, p in model.named_parameters() if name_filter(n)]
    
def init_target(params_iterator_default, params_iterator_target):
    """Initialize target parameters with source parameters."""
    with torch.no_grad():
        for (n, p), p_targ in zip(
                ((n, p) for n, p in params_iterator_default),
                (p for n, p in params_iterator_target)):
            p_targ.data.copy_(p.data)
            assert torch.all(p_targ.data == p.data) and p.requires_grad and not p_targ.requires_grad

# --------------------------
# Success Rate Estimation Model
# --------------------------

class SRFunction(nn.Module):
    """
    Success Rate estimation model using a language model backbone.
    """
    def __init__(self, model, tokenizer, name, target=False):
        super(SRFunction, self).__init__()
        self.device = 'cuda'
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        
        # Value head to predict success probability
        self.value_head_op = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1)
        ).to(self.device)
        
    def forward(self, goals):
        """Forward pass to get success probability predictions for goals using a causal LM."""
        # Tokenize the input and create attention masks.
        tokenized_inputs = self.tokenizer(
            goals, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False, 
            add_special_tokens=True  # Ensure special tokens are added if needed
        )
        input_ids = tokenized_inputs.input_ids.to(self.model.device)
        attention_mask = tokenized_inputs.attention_mask.to(self.model.device)
        
        # For causal LM, no decoder arguments or adapter settings are needed.
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        
        # Extract embedding: use the hidden state of the last token.
        emb = outputs.hidden_states[-1][:, -1, :].to(torch.float32).to(self.model.device)
        
        # Return the success prediction and the embedding.
        return self.value_head_op(emb), emb
        
    def get_parameters_name_filter(self):
        """Get a filter function for this model's parameters."""
        # Removed reference to adapter 'default' as it is not used for causal LLMs.
        return lambda n: f'.{self.name}.' in n or 'value_head_op.' in n or 'default' in n

# --------------------------
# Main Execution Logic
# --------------------------

def main(seed):
    # Configuration parameters
    config = {
        # Paths and run information
        "data_path": "./to_save.pkl",
        "base_path": "./",
        "llm_path": "./LLMs/Qwen2.5-0.5B",
        "run_name": 'scaling_estimation_30k',
        "seed": seed,
        
        # Model configuration
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        
        # Training parameters
        "batch_size": 128,
        "gradient_batch_size": 128,
        "eval_batch_size": 256,
        "learning_rate": 1e-4,
        "buffer_size": 1000,
        "update_interval": 5
    }
    

    # Extract RL data
    data = pickle.load(open(config["data_path"], "rb"))
    rl_goals = data["goals"]
    rl_succes = data["successes"]
    
    # Initialize model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["llm_path"])
    
    # Configure quantization
    q_config = BitsAndBytesConfig(**config["quantization"])
    
    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config["llm_path"],
        quantization_config=q_config,
        device_map="auto",
        output_hidden_states=True
    )
    
    # Prepare model for k-bit training and configure LoRA
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(**config["lora"])
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Ensure LoRA modules are on the correct device
    parent_module_device = None
    for name, param in model.named_modules():
        if "lora_" in name:
            if hasattr(param, "weight"):
                param.to(parent_module_device)
        else:
            if hasattr(param, "weight"):
                parent_module_device = param.weight.device
            else:
                parent_module_device = None
    
    
    # Map keys to goal categories
    category_keys = data["promblems_by_type"]
        
    # Print the number of keys in each category
    for category, category_keys_list in category_keys.items():
        print(f"Number of {category} goals: {len(category_keys_list)}")
    
    # Initialize the SR estimation model
    sr_func = SRFunction(model, tokenizer, name="sr").to(device)
    sr_parameters_filter = sr_func.get_parameters_name_filter()
    
    # Set up training components
    trainable_params = get_filtered_params(sr_func, sr_parameters_filter)
    print_trainable_parameters(
        get_filtered_params(sr_func, sr_parameters_filter, True), 
        "default"
    )
    
    optimizer = torch.optim.Adam(trainable_params, lr=config["learning_rate"])
    
    # Initialize training buffers and metrics
    goal_buffer = deque(maxlen=config["buffer_size"])
    success_buffer = deque(maxlen=config["buffer_size"])
    
    losses = []
    errors = {
        'Algebra': [],
        'Geometry': [],
        'Number Theory': []
    }
    est = {
        'Algebra': [],
        'Geometry': [],
        'Number Theory': []
    }
    
    act = {
        'Algebra': [],
        'Geometry': [],
        'Number Theory': []
    }
    
    # Training loop
    print("Starting training loop...")
    for i, (goal, success) in enumerate(tqdm(zip(rl_goals, rl_succes))):
        # Add data to buffers
        goal_buffer.append(goal)
        success_buffer.append(success)
        
        # Check if an update is needed
        update = i % config["update_interval"] == 0
        
        # Update model if conditions are met
        if len(goal_buffer) > 0 and update:
            # Create importance sampling weights based on recency
            p = np.arange(1, len(goal_buffer) + 1)
            p = p / p.sum()
            
            # Sample batch and prepare data
            idx = np.random.choice(len(goal_buffer), size=config["batch_size"], p=p)
            goals = [goal_buffer[i] for i in idx]
            success_tensor = torch.tensor([success_buffer[i] for i in idx], dtype=torch.float32).unsqueeze(1).to(device)
            
            nb_batches = len(goals) // config["gradient_batch_size"]
            for j in range(0, len(goals), config["gradient_batch_size"]):
                # Forward pass and compute loss
                pred, _ = sr_func(goals[j:j+config["gradient_batch_size"]])
                loss = F.binary_cross_entropy_with_logits(pred, success_tensor[j:j+config["gradient_batch_size"]]) / nb_batches
                # Backward pass and optimization
                loss.backward()
                
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Track losses
            losses.append(loss.item())
            if len(losses) % 100 == 0:
                print(f"Average loss over last 100 updates: {np.mean(losses[-100:]):.4f}")
        
        # Run evaluation when needed
        if i % 1000 == 0:
            print(f"\nEvaluating at step {i}:")
            
            for k, g in category_keys.items():
                # Randomly sample a subset (a batch) of problems from category 'k'
                batch_size_eval = config["eval_batch_size"]
                sample_size = min(batch_size_eval, len(g))
                indices = np.random.randint(0, len(g), size=sample_size)
                sample_batch = [g[idx] for idx in indices]
                
                with torch.no_grad():
                    batch_success, _ = sr_func(sample_batch)
                
                # Apply sigmoid to get probabilities and compute mean success rate
                estimated_success = torch.sigmoid(batch_success)
                estimated_success_mean = estimated_success.mean().item()
                
                # Print results and track errors
                print(f"  {k} estimation: {estimated_success_mean:.4f}, actual: {data['true_sr'][k][i]:.4f}")
                errors[k].append(abs(estimated_success_mean - data['true_sr'][k][i]))
                est[k].append(estimated_success_mean)
                act[k].append(data['true_sr'][k][i])
                
            # Free unused GPU memory
            torch.cuda.empty_cache()
    
    # Print final error statistics
    print("\nFinal estimation errors:")
    for k, err in errors.items():
        if err:
            print(f"  {k}: Mean error = {np.mean(err):.4f}, Max error = {np.max(err):.4f}")
    
    # Save results
    results_path = f"{config['base_path']}/Results/magellan/math_sr_estimation_0_5b/seed_{config['seed']}/"
    os.makedirs(results_path, exist_ok=True)
    with open(results_path + "sr_estimation_qwen05b.pkl", 'wb') as file:
        pickle.dump({
            'errors': errors,
            'losses': losses,
            'est': est,
            'act': act
        }, file)
        
    # Save estimator
    torch.save(trainable_params, results_path + "sr_estimation_qwen05b.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SR estimation training')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args.seed)