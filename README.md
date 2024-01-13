# Open-Instruction-Generalist Dataset

Open Instruction Generalist (OIG) Dataset is intended to train assistants that are part of the LAION-AI's family of assistants.  OIG Assistants will be trained on the OIG dataset, a massive synthetic instructions with the goal of performing many diverse types of tasks. 

We will have several versions of the OIG Assistant dataset ranging from a small (less than 1M) high quality synthetic dataset, to a massive synthetic instruction dataset. The research goal of OIG Assistant is to create high performing bots by using simple finetuning instead of RLHF.

We will create ever larger instruction datasets with the goal to generate eventually 1T medium quality tokens of instructions. The receipe for training is to do additional pretrain on some subset of the larger instruction sets, followed by a finetune on OIG-small or some other high quality small dataset.

* OIG-small-chip2 (200K) - Done and released. See  small_instruction_set sub-directory.
* OIG-40M - Done - Done and released. See 40M sub-directory

# OIG-Moderation

We have also created a small subset of safety data to tag instructions for moderation. This dataset was created by volunteers and also curated and augmented from public datasets (see https://huggingface.co/datasets/ontocord/OIG-moderation)

* OIG_safety_v0.1.jsonl (66K)
* OIG_safety_v0.2.jsonl (134K)

## Related Projects
* Check out LAION's [Open Assistant Project](https://github.com/LAION-AI/Open-Assistant). We aim to build a chatbot based on RLHF and human feedback data.
* Check out our friends Together.xyz's [OpenChatKit](https://github.com/togethercomputer/OpenChatKit). They trained a bot based on OIG!

## Models
The community has trained several models based on a subset of the OIG datasets including:

- Rallio67/joi2_(20,12,7)B_instruct_alpha
- Rallio67/chip2_(20,12,7)B_instruct_alpha
- Rallio67/joi_(20,12,7)B_instruct_alpha
- Rallio67/chip_(20,12,7)B_instruct_alpha
- togethercomputer/GPT-NeoXT-Chat-Base-20B

### Safety models

- SummerSigh/T5-Base-Rule-Of-Thumb
- SummerSigh/Safety-Policy
- SummerSigh/BART-Base-Rule-Of-Thumb
- shahules786/prosocial-classifier
- shahules786/Safetybot-mt5-base
- shahules786/Safetybot-T5-base
- togethercomputer/GPT-JT-Moderation-6B

Available on huggingface.co. 
