

# This is the Open Instruction Generalist Dataset

- See https://huggingface.co/datasets/laion/OIG

This is our attempt to create a large instruction dataset of medium quality along with a smaller high quality instruciton dataset (OIG-small-chip2).

The data is in the form of jsonl objects, with at least a 'text' field. Some datasets may also include a 'metadata' field. The 'text' field contains a string of the form of one or more of:

- \<human\>: instruction\n\<bot\>: response
- \<human\>: instruction\n\<bot\>: response .. \<human\>: instruction\n\<bot\>: response
  
The purpose of the larger dataset is to perform continued pre-training, followed by a finetune on the smaller high quality dataset.

The purpose of the smaller OIG-small-chip2 dataset is to make it easy to convert a language model pretrained on large amounts of text into an instruction following model using a small amount of additional compute via finetuning or softprompt tuning.

Many additional datasets are being prepared by various community members and will be incorporated into this dataset as we are able to verify the quality and formatting of the data. Our goal is to make helpful and non-toxic instruction tuned models available to everyone.

OIG is currently at 44M. We will continue to publish ever larger diverse instruction datasets with the goal of creating 1 trillion tokens of diverse instructions - enough to pretrain an LLM from scratch.

WIP: Explanations of the following datasets will be provided. 

## unified_abstract_infill.jsonl (~232000)
## unified_basic.jsonl (30)
## unified_conv_finqa.jsonl (~9000)
## unified_cuad.jsonl (~500)

## unified_essays.jsonl (~2000)
- essays available on the public web 
## unified_grade_school_math_instructions.jsonl (~9000)
- https://github.com/openai/grade-school-math
## unified_hc3_human.jsonl (~58000)
## unified_image_prompts_instructions.jsonl (~15000)
- A very small subset of LAION-400M
## unified_joke_explanations.jsonl (356)
- Crawled from public internet. 
## unified_mathqa_flanv2_kojma_cot.jsonl (~107000)
- https://huggingface.co/datasets/math_qa, 
## unified_merged_code_xp3.jsonl (~67000)
-  https://huggingface.co/datasets/bigscience/xP3 
## unified_multi_news.jsonl (~90000)
- https://www.tensorflow.org/datasets/catalog/multi_news
## unified_multi_sum.jsonl (~1700000)
## unified_nq.jsonl (~307000)
## unified_openai_summarize_tldr.jsonl (~233000)
- https://github.com/openai/summarize-from-feedback
## unified_oscar_en_sample_dialog.jsonl (~2670000)
- https://oscar-project.org/
- https://huggingface.co/datasets/TurkuNLP/register_oscar
## unified_plot_screenplay_books_dialog.jsonl (~8000)
-  https://github.com/markriedl/WikiPlots extracted from Wikipedia, snippets from the Pile’s https://huggingface.co/datasets/the_pile_books3, and snippets of screenplays available on the public web.
## unified_sqlv1.jsonl (~17000)
- public text 2 sql datasets.
## unified_sqlv2.jsonl(~24000)
- public text 2 sql datasets.
## unified_squad_v2.jsonl (~19000)
-  https://rajpurkar.github.io/SQuAD-explorer/
## unified_squad_v2_more_neg.jsonl (~19000)
-  https://rajpurkar.github.io/SQuAD-explorer/
## unified_ul2_plus_oscar_en_sample_dialog.jsonl (~2900000)
- https://oscar-project.org/
- https://huggingface.co/datasets/TurkuNLP/register_oscar
## unified_unifiedskg_instructions.jsonl (~223000)
- https://github.com/HKUNLP/UnifiedSKG 
## unified_unnatural_instructions.jsonl (~238000)
-  https://github.com/orhonovich/unnatural-instructions
## unified_xp3_sample.jsonl (~188000)
-  https://huggingface.co/datasets/bigscience/xP3 
## unified_canadian_parliament.jsonl(~301000)
- https://openparliament.ca/data-download/
## unified_poetry_2_song.jsonl (~12000)
- https://huggingface.co/datasets/merve/poetry
- https://huggingface.co/datasets/matthh/gutenberg-poetry-corpus 
## unified_flan.jsonl (~2700000)
-  https://github.com/google-research/FLAN/tree/main/flan/v2 
## unified_ni.jsonl (~256000)
 - https://github.com/allenai/natural-instructions 
## unified_p3.jsonl (~31000000)
- https://huggingface.co/datasets/bigscience/P3 
## unified_soda_dialog.jsonl (~1200000)
-  https://huggingface.co/datasets/allenai/soda
## unified_rallio_soda_upgraded_2048.jsonl (~210000)
-  https://huggingface.co/datasets/allenai/soda 
- a newer version of the unified_soda_dialog dataset, with multiple dialogs on one line
- recommend to use either the unified_soda_dailog.jsonl or unified_rallio_soda_upgraded_2048, and not both.
## unified_rallio_safety_and_prosocial.jsonl (~319000)
- Generated from public datasets and generated from Wiki similar to the chip2 data
- Find a full list in the end of the document
- This dataset also includes https://huggingface.co/datasets/allenai/prosocial-dialog and https://huggingface.co/datasets/Anthropic/hh-rlhf  
## unified-chip2.jsonl / OIG-small-chip2 (~210000):
This dataset was created as part of the LAION OA effort by @rallio67 and other members of the LAION contributors. It is a high quality dataset intended to be mixed into a large pre-train dataset and can be used for a final finetune. Chip2 contains: 

### Python Code Examples (~6,000):
A set of instruction / response pairs where the User requests the agent to generate a python function. These examples were generated using a large language model and few shot prompting with python code verified to execute. There are also ~3000 examples of manually curated one line python code examples from the Conala publication (see: https://conala-corpus.github.io/)

### Natural Instruction Examples (~124,000):
A balanced set of diverse natural and factual questions and answers made using few shot prompted UL2 20B and an instruction tuned GPT-NeoX-20B model (Chip) and then rejection sampled using multiple automatic evaluations to remove low quality outputs and to filter out factually inaccurate answers. Also includes some filtered natural instructions from Anthropic Helpful instructions (see: https://github.com/anthropics/hh-rlhf).

### Generic Harmless Instruction Examples (~6,500):
A set of instruction / response pairs sourced from the Anthropic redteam paper github (see: https://github.com/anthropics/hh-rlhf). This dataset includes a lot of data regarding real humans trying to make the Anthropic language models say harmful/toxic/trolling things. For this dataset only examples that were rated lowly on the harmful scale (0,1,2 out of 4, where 4 is the most toxic) were included. Again, only the first lines of dialogue (instruction, first_agent_response) were retained.

### Instruction/Responses with Lists (~14,000):
A set of filtered and reformatted instruction / response pairs where the agent response contains a list. Sourced from the Anthropic github (see: https://github.com/anthropics/hh-rlhf). Sourced from wikihow text lists created by b-mc2 (https://huggingface.co/datasets/b-mc2/wikihow_lists). And rejection filtered instruction response pairs generated by Chip20B that contained lists. All lists are formatted in a similar style.

### Follow-up questions (~12,500):
Examples of instructions and responses where an appropriate response is to ask for more information from the prompter. These examples were generated from a combination of few shot prompted UL2 20B (to generate natural questions) and a large dialogue prompted language model to generate the responses containing follow-up questions.

### Wikipedia Toxic Adversarial Questions (~12,000):
Questions and answers generated from wikipedia articles that discuss potentially sensitive topics (flagged as potentially toxic by an early toxicity detection model).

### Grade School Math GSM8K (~9,000):
GSM8K is a dataset of 8.5K high quality linguistically diverse grade school math word problems created by human problem writers. The dataset is segmented into 7.5K training problems and 1K test problems. These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer. A bright middle school student should be able to solve every problem. It can be used for multi-step mathematical reasoning. (https://github.com/openai/grade-school-math)

### Reasoning Instructions (~4,500):
Examples from the Com2Sense and Strategy QA datasets that were reformatted into natural instructions using large language models with few shot prompting and additional quality filtering steps.

### Character and Scene Descriptions (~30,000):
Examples of instructions and responses for the generation of character or scene descriptions. Scenes were sourced from video game wikis and reformatted into instruction / response format using large language models or generated by few shot prompting with large language models.

## Support this project
Your contributions and feedback support the open source ecosystem, improve the bot and provide datasets for future AI research. To participate you can:

Submit Github issues, track issues and help create datasets that need improvement. https://github.com/LAION-AI/Open-Instruction-Generalist 
Join our Discord to talk with other team members working on this! https://discord.gg/xBPBXfcFHd 

## Disclaimer
These datasets contain synthetic data and in some cases data that includes humans trying to get the language model to say toxic/offensive/trolling things. If you are concerned about the presence of this type of material in the dataset please make sure you carefully inspect each of the entries and filter appropriately. Our goal is for the model to be as helpful and non-toxic as possible and we are actively evaluating ways to reduce or eliminate undesirable content from the instruction tuning datasets.

## License
The OIG dataset that is authored by LAION volunteers is released under an Apache 2.0 license. However, the data also includes content licensed under other permissive licenses such as Wikipedia data which is licensed under CC-BY-SA, or web-crawled data which is used under fair use principles.

## Acknowledgement
- We would like to thank all of our amazing LAION volunteers including: @Rallio, @Jue, @Ce Zhang, @Player-1, @Laurel, @danielpatrickhug, @Jjmachan, @Mylo, @Khalid, @Coco.han, @Jordiclive, @Pszemraj, and many others.
- We would like to thank Together for their tireless dedication to the open source and AI community and their contribution to many of the datasets.
- We would like to thank AI Horde and user @Db0 for their incredible contribution of filtered data that were flagged as unethical.
- Lastly, Ontocord.ai’s founders are grateful to have the opportunity to create a portion of the data augmentation and safety-moderation code for this project.

