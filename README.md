# Open-Instruction-Generalist Dataset

Open Instruction Generalist (OIG) Dataset is intended to train assistants that are part of the LAION-AI's OpenChat family of assistants.  OIG Assistants will be trained on the OIG dataset, a massive synthetic instructions with the goal of performing many diverse types of tasks. 

We will have several versions of the OIG Assistant ranging from an OIG Assistant that is trained on a small (less than 1M) high quality synthetic dataset, to an OIG Aisstant trained on medium quality but massive synthetic instructions. The research goal of OIG Assistant is to create high performing bots by using simple finetuning instead of RLHF.

We will create ever larger instruction datasets with the goal to generate eventually 1T medium quality tokens of instructions. The receipe for training is to do additional pretrain on some subset of the larger instruction sets, followed by a finetune on OIG-small or some other high quality small dataset.

* OIG-small-chip2 (200K) - Done and released. See  small_instruction_set sub-directory.
* OIG-40M - Done - Done and released. See 40M sub-directory
* OIG-100M - In progress, to be released expected March 30, 2023

# OIG-Moderation

We have also created a small subset of safety data to tag instructions for moderation. This dataset was created by volunteers and also curated and augmented from public datasets (see https://huggingface.co/datasets/ontocord/OIG-moderation)

* OIG_safety_v0.1.jsonl (66K)
* OIG_safety_v0.2.jsonl (134K)
