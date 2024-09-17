# SFT Red Teaming Evaluation Script

## Overview

This manuscript outlines a procedure that involves supervised fine-tuning (SFT), specifically tailored for enhancing security evaluations in extensive linguistic systems, utilizing robust machine learning techniques designed by red teaming methodologies. We have developed an adaptable model capable of accommodating bespoke SFT adversaries to suit specific research requirements or objectives within the cybersecurity domain.

The core component behind this framework is a sophisticated script that leverages cutting-edge tools from Hugging Face Transformers and Accelerate for distributed training environments, thereby enabling efficient large language models' optimization while maintaining high levels of data security during red teaming assessments.
## Usage

To run the script, use the following command structure:

```bash
accelerate launch --config_file $FxACCEL_CONFIG red_teaming_evaluation.py --args {YOUR ARGUMENTS}
```

## Options

The script accepts the following command-line arguments:

- `--model_name`, `-mn`: Name of the model to be fine-tuned (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `--model_type`, `-mt`: Type of the model (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `--save_model_name`, `-smn`: Name to save the fine-tuned model as (default: "saved_model")
- `--scheduler_type`, `-st`: Type of learning rate scheduler (default: "none")
- `--num_warmup_steps`, `-nws`: Number of warmup steps for the scheduler (default: 0)
- `--batch_size`, `-bs`: Batch size for training (default: 8). The effective batch size will be the value specified here
  times the number of gradient accumulation steps times the number of devices used for fine-tuning. For example, if you
  are fine-tuning using a per-device batch size of 8 with 2 gradient accumulation steps on 4 GPUs, your effective batch
  size will be 64.
- `--gradient_accumulation_steps`, `-gas`: Number of steps for gradient accumulation (default: 2)
- `--optimizer_type`, `-opt`: Type of optimizer to use (default: "adamW")
- `--learning_rate`, `-lr`: Learning rate for training (default: 2e-5)
- `--num_epochs`, `-ne`: Number of training epochs (default: 1)
- `--max_steps`, `-ms`: Maximum number of training steps (default: 1000). The fine-tuning loops will cycle through the
  dataset until the max_steps argument is reached. This behavior can be modified in training.py by changing the loop
  ordering.
- `--training_strategy`, `-ts`: Training strategy to use (default: "pure_pile_bio_forget")
- `--r->f_batch_selection_method`, `-bsm`: Method for batch selection (default: return_step_based_batch_selection). The
  other option is return_coin_flip_batch_selection. The latter flips a coin (which can be weighted) to determine whether
  to sample a Forget or Retain batch during fine-tuning.
- `--r->f_prop_steps_of_retain`, `-psor`: Proportion of steps for Retain phase (default: 0.4). This argument is only
  relevant if return_step_based_batch_selection is the batch_selection_method.
- `--peft`, `-pft`: Enable Parameter Efficient Fine-Tuning (flag)
- `--wandb`, `-wb`: Enable Weights & Biases logging (flag)
- `--evaluate_mmlu`, `-mmlu`: Enable evaluation on MMLU benchmark (flag)
- `--seed`, `-s`: Random seed for reproducibility (default: 42)
