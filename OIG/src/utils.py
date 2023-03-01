#@title Utils
"""
Copyright 2023, LAION contributors, inclduing Ontocord, LLC
and the other authors of OIG

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License. 
"""
import os
try:
  import datasets, transformers
except:
  import os
  import nltk
  nltk.download('punkt')
  os.system("pip install datasets  transformers bitsandbytes accelerate sentencepiece")
  os.system("pip install spacy==2.1.8")
  os.system("pip install scispacy==0.2.3")
  os.system("pip install blackstone==0.1")
  os.system("pip install https://blackstone-model.s3-eu-west-1.amazonaws.com/en_blackstone_proto-0.0.1.tar.gz")
  os.system("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_ner_craft_md-0.2.0.tar.gz")
  os.system("python -m spacy download en_core_web_sm")
  
#RESTART THE RUNTIME

from transformers import AutoTokenizer, OPTForCausalLM,  AutoModelForCausalLM, AutoModel, T5Tokenizer, T5PreTrainedModel
from transformers import AutoTokenizer,  AutoModelForCausalLM, AutoModel
from transformers import T5Tokenizer, T5EncoderModel, AutoModel
from transformers import T5PreTrainedModel, T5EncoderModel
from transformers import AutoModelForSeq2SeqLM
from torch import nn
import torch
#from duckduckgo_search import ddg
from transformers import AutoTokenizer, AutoModelForCausalLM
from subprocess import call
import os
#os.system("nvidia-smi --format=csv --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free > __gpu_stats.csv")  
#gpu_memory = int(open("__gpu_stats.csv").read().split("\n")[1].split(",")[3].strip().split()[0])*100000


import os
import spacy

import itertools
import logging
from typing import Optional, Dict, Union
import os
from nltk import sent_tokenize

import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


try:
  if basic_nlp is None: pass
except:
  basic_nlp = spacy.load('en_core_web_sm')
  sci = spacy.load("en_ner_craft_md")
  blackstone = spacy.load("en_blackstone_proto")
  # add the other scispacy ner

import math
import pickle
import time

import torch
# load all needed libraries
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel

import json
  

logger = logging.getLogger(__name__)
# adapted from https://github.com/patil-suraj/question_generation which is under the MIT License
class QGPipeline:
    """Poor man's QG pipeline"""
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ans_model: PreTrainedModel,
        ans_tokenizer: PreTrainedTokenizer,
        qg_format: str,
        device: str,
        default_answers = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.ans_model = ans_model
        self.ans_tokenizer = ans_tokenizer

        self.qg_format = qg_format
        self.default_answers = default_answers
        self.device = device
        if self.model.device != self.device:
            self.model.to(self.device).eval()
            if device == "cpu":
                self.model = torch.quantization.quantize_dynamic(self.model.float(), {torch.nn.Linear}, dtype=torch.qint8)
            else:  
                self.model = self.model.half().to(device)

        if self.ans_model is not self.model:
            if self.ans_model.device != self.device:
                self.ans_model.to(self.device).eval()
                if device == "cpu":
                    self.ans_model = torch.quantization.quantize_dynamic(self.ans_model.float(), {torch.nn.Linear}, dtype=torch.qint8)
                else:  
                    self.ans_model = self.ans_model.half().to(device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"

    def __call__(self, inputs: str, **generate_kwargs):
        self.model.eval()
        self.ans_model.eval()
        ret = []
        with torch.no_grad():
          
          if type(inputs) is str:
            inputs = [inputs]
          default_answers=[]
          if 'default_answers' in generate_kwargs:
            default_answers = generate_kwargs['default_answers']
            if default_answers and type(default_answers[0]) is str:
              default_answers = [default_answers] * len(inputs)
          if len(default_answers) < len(inputs):
            default_answers.extend([[]]*(len(inputs)-len(default_answers)))
          #TODO - we could do in batches that is approximately N words to maximize GPU usage
          for input, default_answer in zip(inputs, default_answers):
            qg_examples = []
            input = " ".join(input.split())
            sents, answers = self._extract_answers(input)
            if self.default_answers:
              answers.append(self.default_answers)
            if default_answer:
              answers.append(default_answer)
            flat_answers = list(itertools.chain(*answers))
            
            if len(flat_answers) == 0:
              ret.append([])
              continue
            answers = [flat_answers]*len(sents) # multi-way q/a
            if self.qg_format == "prepend":
                qg_examples.extend(self._prepare_inputs_for_qg_from_answers_prepend(inputs, answers))
            else:
                qg_examples.extend(self._prepare_inputs_for_qg_from_answers_hl(sents, answers))
            if  qg_examples:
              qg_inputs = [example['source_text'] for example in qg_examples]
              questions = self._generate_questions(qg_inputs)
              output = list(set([(example['answer'], que) for example, que in zip(qg_examples, questions)]))
              ret.append([{'answer': answer, 'question': que} for answer, que in output])
            else:
              ret.append([])
        return ret
    
    def _generate_questions(self, inputs):
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
            num_beams=4,
        )
        
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions
    
    def _extract_answers(self, context):
        sents, inputs = self._prepare_inputs_for_ans_extraction(context)
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        self.ans_model.eval()
        with torch.no_grad():
          outs = self.ans_model.generate(
              input_ids=inputs['input_ids'].to(self.device), 
              attention_mask=inputs['attention_mask'].to(self.device), 
              max_length=32,
          )
        
        dec = [self.ans_tokenizer.decode(ids, skip_special_tokens=False) for ids in outs]
        answers = [item.replace("<pad>","").replace("  ", " ").strip().split('<sep>') for item in dec]
        answers = [i[:-1] for i in answers if i !=[]]
        
        return sents, answers
    
    def _tokenize(self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
    def _prepare_inputs_for_ans_extraction(self, text):
        sents = sent_tokenize(text)

        inputs = []
        for i in range(len(sents)):
            source_text = "extract answers:"
            for j, sent in enumerate(sents):
                if i == j:
                    sent = "<hl> %s <hl>" % sent
                source_text = "%s %s" % (source_text, sent)
                source_text = source_text.strip()
            
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            inputs.append(source_text)

        return sents, inputs
    
    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for i, answer in enumerate(answers):
            if len(answer) == 0: continue
            for answer_text in answer:
                sent = sents[i]
                sents_copy = sents[:]
                
                answer_text = answer_text.strip()
                if answer_text.lower() not in sent.lower(): continue
                ans_start_idx = sent.lower().index(answer_text.lower())
                
                sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
                sents_copy[i] = sent
                
                source_text = " ".join(sents_copy)
                source_text = f"generate question: {source_text}" 
                if self.model_type == "t5":
                    source_text = source_text + " </s>"
                
                inputs.append({"answer": answer_text, "source_text": source_text})
        
        return inputs
    
    def _prepare_inputs_for_qg_from_answers_prepend(self, context, answers):
        flat_answers = list(itertools.chain(*answers))
        examples = []
        for answer in flat_answers:
            source_text = f"answer: {answer} context: {context}"
            if self.model_type == "t5":
                source_text = source_text + " </s>"
            
            examples.append({"answer": answer, "source_text": source_text})
        return examples

    
class MultiTaskQAQGPipeline(QGPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self, inputs: Union[Dict, str], **generate_kwargs):
        if type(inputs) in (list, str):
            # do qg
            return super().__call__(inputs, **generate_kwargs)
        else:
            # do qa
            return self._extract_answer(inputs["question"], inputs["context"], **generate_kwargs)
    
    def _prepare_inputs_for_qa(self, question, context):
        source_text = f"question: {question}  context: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        return  source_text
    
    def _extract_answer(self, question, context):
        source_text = self._prepare_inputs_for_qa(question, context)
        inputs = self._tokenize([source_text], padding=False)
    
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=16,
        )

        answer = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        return answer


class E2EQGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str,
    ) :

        self.model = model
        self.tokenizer = tokenizer

        self.device = device
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"
        
        self.default_generate_kwargs = {
            "max_length": 256,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
    
    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        # TODO: when overrding default_generate_kwargs all other arguments need to be passsed
        # find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        
        input_length = inputs["input_ids"].shape[-1]
        
        # max_length = generate_kwargs.get("max_length", 256)
        # if input_length < max_length:
        #     logger.warning(
        #         "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
        #             max_length, input_length
        #         )
        #     )

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions
    
    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
    
    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs


SUPPORTED_TASKS = {
    "question-generation": {
        "impl": QGPipeline,
        "default": {
            "model": "valhalla/t5-small-qg-hl" ,
            "ans_model": "valhalla/t5-small-qa-qg-hl" ,
        }
    },
    "multitask-qa-qg": {
        "impl": MultiTaskQAQGPipeline,
        "default": {
            "model": "valhalla/t5-small-qa-qg-hl" ,
        }
    },
    "e2e-qg": {
        "impl": E2EQGPipeline,
        "default": {
            "model": "valhalla/t5-small-e2e-qg" ,
        }
    }
}

def pipeline(
    task: str,
    model: Optional = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    qg_format: Optional[str] = "highlight",
    ans_model: Optional = None,
    ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    device: str = "cpu",
    **kwargs,
):

    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]
    models_same=False
    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"]
    if ans_model is None:
        ans_model = targeted_task["default"].get("ans_model", None)
    if isinstance(model, str) and isinstance(ans_model, str) and model == ans_model:
      models_same = True
    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )
    
    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = AutoTokenizer.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            #print(tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

                
    
    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model).eval()
        if device == "cpu":
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        else:
            model = model.half().to(device)
                
    if task == "question-generation":
        if ans_model is None:
            # load default ans model
            ans_model = targeted_task["default"]["ans_model"]
            ans_tokenizer = AutoTokenizer.from_pretrained(ans_model)
            if models_same:
              ans_model = model
            else:
              ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model).eval()
              if device == "cpu":
                ans_model = torch.quantization.quantize_dynamic(ans_model, {torch.nn.Linear}, dtype=torch.qint8)
              else:
                ans_model = ans_model.half().to(device)
                
        else:
            # Try to infer tokenizer from model or config name (if provided as str)
            if models_same:
              ans_tokenizer = tokenizer
            elif ans_tokenizer is None:
                if isinstance(ans_model, str):
                    ans_tokenizer = ans_model
                else:
                    # Impossible to guest what is the right tokenizer here
                    raise Exception(
                        "Impossible to guess which tokenizer to use. "
                        "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                    )
            
            # Instantiate tokenizer if needed
            if isinstance(ans_tokenizer, (str, tuple)):
                if isinstance(ans_tokenizer, tuple):
                    # For tuple we have (tokenizer name, {kwargs})
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer[0], **ans_tokenizer[1])
                else:
                    ans_tokenizer = AutoTokenizer.from_pretrained(ans_tokenizer)

            if models_same:
              ans_model = model
            elif isinstance(ans_model, str):
                ans_model = AutoModelForSeq2SeqLM.from_pretrained(ans_model).eval()
                if device == "cpu":
                    ans_model = torch.quantization.quantize_dynamic(ans_model, {torch.nn.Linear}, dtype=torch.qint8)
                else:
                    ans_model = ans_model.half().to(device)
    
    if task == "e2e-qg":
        return task_class(model=model, tokenizer=tokenizer, device=device)
    elif task == "question-generation":
        return task_class(model=model, tokenizer=tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer, qg_format=qg_format, device=device)
    else:
        return task_class(model=model, tokenizer=tokenizer, ans_model=model, ans_tokenizer=tokenizer, qg_format=qg_format, device=device)

class T5EncoderWithProjection(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.t5_encoder = T5EncoderModel(config)
        self.projection = nn.Linear(config.d_model, config.d_model, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **input_args):
        hidden_states = self.t5_encoder(**input_args).last_hidden_state
        hidden_states = hidden_states[:, 0, :]
        batch_embeddings = self.projection(hidden_states)
        return batch_embeddings


def encode_rankgen(inputs, vectors_type="prefix", device='cuda', **model_args):
    if isinstance(inputs, str):
        inputs = [inputs]
    if vectors_type == 'prefix':
        inputs = ['pre ' + input for input in inputs]
    else:
        inputs = ['suffi ' + input for input in inputs]
    input_ids = rankgen_tokenizer(inputs, padding=True, return_tensors="pt").to(device)
    for key, val in model_args.items():
      input_ids[key] = val
    with torch.no_grad():
      batch_embeddings = rankgen_model(**input_ids)
    return batch_embeddings


def run_model(input_string, model, tokenizer, device='cuda', **generator_args):
  with torch.no_grad():
    input_ids = tokenizer(input_string, padding=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    input_ids['no_repeat_ngram_size']=max(generator_args.get('no_repeat_ngram_size',4), 4)
    input_ids['do_sample']=True
    input_ids['top_p']=True
    input_ids['do_sample']=0.95
    input_ids['penalty_alpha']=0.6
    input_ids['top_k']=10
    if 'galactica' in tokenizer.name_or_path and 'token_type_ids' in input_ids:
      del input_ids['token_type_ids']
    for key, val in generator_args.items():
      input_ids[key] = val    
    res = model.generate(**input_ids)
    return [ret.replace("..", ".").replace(".-", ".").replace("..", ".").replace("--", "-").replace("--", "-") for ret in tokenizer.batch_decode(res, skip_special_tokens=True)]

#Mean Pooling - Take attention mask into account for correct averaging
#TODO, mask out the prefix for data that isn't the first portion of a prefixed text.
def mean_pooling(model_output, attention_mask):
    with torch.no_grad():
      token_embeddings = model_output.last_hidden_state
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_ext(para, model, tokenizer, return_answer_only=True, do_self_contrastive=True, max_length=512, max_return_sequences=4, ret=None, do_sample=True, do_beam=False, device="cuda", target_lang=None):
    para = para.strip()
    input_ids = tokenizer.encode(para, return_tensors='pt')
    input_ids = input_ids.to(device)
    if ret is None: ret = {}
    with torch.no_grad():
      if do_sample:
          # Here we use top_k / top_k random sampling. It generates more diverse queries, but of lower quality
          outputs = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                no_repeat_ngram_size=4, 
                do_sample=True,
                top_p=0.95,
                penalty_alpha=0.6 if do_self_contrastive else None, 
                top_k=10, 
                num_return_sequences=max(1, int(max_return_sequences/2)) if do_beam else max_return_sequences
                )
        
          for i in range(len(outputs)): # can use batch_decode, unless we want to do something special here
            query = tokenizer.decode(outputs[i], skip_special_tokens=True)
            if return_answer_only:
              query = query[len(para):].lstrip(".? \n\t")
            ret[query] = 1

      if do_beam:

        # Here we use Beam-search. It generates better quality queries, but with less diversity
          outputs = model.generate(
                input_ids=input_ids, 
                max_length=max_length, 
                num_beams=max(int(max_return_sequences/2) if do_sample else max_return_sequences,5), 
                no_repeat_ngram_size=4,
                penalty_alpha=0.6 if do_self_contrastive else None,  
                num_return_sequences=max(1, int(max_return_sequences/2)) if do_sample else max_return_sequences, 
                early_stopping=True
            )
    

          for i in range(len(outputs)): # can use batch_decode, unless we want to do something special here
            query = tokenizer.decode(outputs[i], skip_special_tokens=True)
            if return_answer_only:
              query = query[len(para):].lstrip(".? \n\t")
            ret[query] = 1

    return list(ret.keys())  

def run_python_and_return(s):
  try:
    ret = {'__ret': None}
    exec(s, ret)
    return ret['__ret']
  except:
    return ''

#poorman's reverb. TODO: we need to use semantic matching of relationship to paragraph to filter out bad relationships.
def get_verb_relation(text):
  doc = basic_nlp(text)
  verb_relationship = ""
  orig_verb = ""
  for token in doc:
    #print (token, token.tag_)
    if token.tag_.startswith("VB") and token.tag_ not in {"VBZ", } and token.lemma_ not in {'do', 'be', 'have'}:
      orig_verb = token.text
      verb_relationship = str(token.lemma_)
      continue  
    if verb_relationship:
      if token.tag_ == "IN":
        orig_verb += " "+token.text
        verb_relationship += "_"+str(token.lemma_)
        break
      else:
        break
  return verb_relationship, orig_verb

#need to filter out rel that don't match embedding of full text. these are spurious
def ner_rel_template_extract(text, min_ner_len=5, length_for_rel=50):
  ret = {}
  orig_text = text
  text2 = text.replace("{", "-lbracket-").replace("}", "-rbracket-")
  ner_cnt = {}
  for nlp in [blackstone, sci, basic_nlp]:
    doc =nlp(text)
    ents = [(ent.text.strip(), ent.label_) for ent in  list(doc.ents) if len(ent.text.strip()) >= min_ner_len]
    ents.sort(key=lambda a: len(a[0]), reverse=True)
    for st, label in ents:
      #we are not doing NER for code
      if "->" in st or "{" in st or "}" in st: continue 
      if st in text:
        ner_cnt[label] = ner_cnt.get(label, -1)
        ner_cnt[label] += 1
        if ner_cnt[label] > 0:
          text2 = text2.replace(st,'{'+label+'_'+str(ner_cnt[label])+'}')
          ret[st] = label+'_'+str(ner_cnt[label])
        else:
          text2 = text2.replace(st,'{'+label+'}')
          ret[st] = label
        text = text.replace(st,' ')
    rels =[]
    if nlp == basic_nlp:

      args = dict([(b, "{"+a+"}") for a, b in ret.items() ])
      if args:
        print (args, '**', text2)
        text3 = text2.format(**args)
        text4 = text3.replace("{", " ").replace("}", " ")
        for entity in ret.keys():
          if "{"+entity+"}" not in text3:
            print ('problem', "{"+entity+"}", '***', text3)
          text5= text4[text3.index("{"+entity+"}"):]
          if len(text5) > length_for_rel:
            text5 = text5[:length_for_rel]
          rel, orig_verb = get_verb_relation(text5)
          if rel:
            text6 = text3[text3.index("{"+entity+"}"):].split(orig_verb)[1]
            if "{" in text6:
              text6 = text6.split("{",1)[1]
              if "}" in text6:
                entity2 = text6.split("}")[0]
                rels.append ((entity.replace(" ", "_") ,rel, entity2.replace(" ", "_") ))
      
  return ret, text2.replace("-lbracket-", "{").replace("-rbracket-", "}"), rels


#example few shot instruction generation with gpt (JT is better at this)
few_shot_query_to_instruction = """
Convert the given query into an instruction.

Input: What are the famous places we should not miss in Paris?
Output: So, I'm going to Paris next week. List a bunch of places to visit.
##
Input: Explain quantum physics like I'm 5 years old.
Output: Assume you are speaking to a five year old. Explain step by step, quantum physics and why it's important to know.
##
Input: Why do people like to travel to the Galapagos?
Output: We're taking a trip to the Galapagos Islands soon. Find out why people travel to the Galapagos.
##
"""
import itertools


def generate_instructions_from_query(inputs, model, tokenizer):
  out = run_model([few_shot_query_to_instruction + "Input: "+instr for instr in inputs], model, tokenizer, max_length=256)
  prefix_len = len(few_shot_query_to_instruction + "Input: ")
  out = [a[prefix_len:].split("#")[0].split('Output',1) for a in out]
  out = [(a_b[0].strip("\n: "),a_b[1].strip("\n: ")) for a_b in out if type(a_b) is list and len(a_b) > 1]
  return (out)

def generate_query_and_instructions(model, tokenizer, max_return_sequences=3):
  out = generate_ext(few_shot_query_to_instruction + "Input:", model, tokenizer, return_answer_only=False,  max_length=256, max_return_sequences=max_return_sequences)
  prefix_len = len(few_shot_query_to_instruction + "Input:")
  out = [a[prefix_len:].split("#")[0].split('Output',1) for a in out]
  out = [(a_b[0].strip("\n: "),a_b[1].strip("\n: ")) for a_b in out if type(a_b) is list and len(a_b) > 1]
  return (out)

#add "Input:" or "Input: {query} \nOutput:"

### Adapted from https://github.com/Rallio67/language-model-agents which is under the Apache2 licenese


# This notebook will run on a system with a single RTX3090 (24 GB vram).
# You need to install accelerate, bitsandbytes, and transformers

# This device map will work a GPU with > 24GB vram.
# It uses nearly all the memory.
device_map_T5_13B = {
    "shared": 0,
    "decoder.embed_tokens": 0,
    "encoder.embed_tokens": 0,
    "encoder.block.0": 0,
    "encoder.block.1": 0,
    "encoder.block.2": 0,
    "encoder.block.3": 0,
    "encoder.block.4": 0,
    "encoder.block.5": 0,
    "encoder.block.6": 0,
    "encoder.block.7": 0,
    "encoder.block.8": 0,
    "encoder.block.9": 0,
    "encoder.block.10": 0,
    "encoder.block.11": 0,
    "encoder.block.12": 0,
    "encoder.block.13": 0,
    "encoder.block.14": 0,
    "encoder.block.15": 0,
    "encoder.block.16": 0,
    "encoder.block.17": 0,
    "encoder.block.18": 0,
    "encoder.block.19": 0,
    "encoder.block.20": 0,
    "encoder.block.21": 0,
    "encoder.block.22": 0,
    "encoder.block.23": 0,
    "encoder.final_layer_norm": 0,
    "encoder.dropout": 0,
    "decoder.block.0": 0,
    "decoder.block.1": 0,
    "decoder.block.2": 0,
    "decoder.block.3": 0,
    "decoder.block.4": 0,
    "decoder.block.5": 0,
    "decoder.block.6": 0,
    "decoder.block.7": 0,
    "decoder.block.8": 0,
    "decoder.block.9": 0,
    "decoder.block.10": 0,
    "decoder.block.11": 0,
    "decoder.block.12": 0,
    "decoder.block.13": 0,
    "decoder.block.14": 0,
    "decoder.block.15": 0,
    "decoder.block.16": 0,
    "decoder.block.17": 0,
    "decoder.block.18": 0,
    "decoder.block.19": 0,
    "decoder.block.20": 0,
    "decoder.block.21": 0,
    "decoder.block.22": 0,
    "decoder.block.23": 0,
    "decoder.final_layer_norm": 0,
    "decoder.dropout": 0,
    "lm_head": 0,
}




# Sort_Tuple sorts a list of tuples
# by the second element.
def Sort_Tuple(tup):
    tup.sort(key=lambda x: x[1], reverse=True)
    return tup


# ask_flan_T5 takes a text input and returns the
# response of FLAN_T5 and a normalized logits
# score for the generation.
def ask_flan_T5(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt").cuda(0)
    outputs = model.generate(
        inputs,
        do_sample=True,
        top_p=0.95,
        eos_token_id=1,
        max_new_tokens=50,
        bos_token_id=0,
        temperature=0.9,
        return_dict_in_generate=True,
        output_scores=True,
    )
    out_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    probs = torch.stack(outputs.scores, dim=1).softmax(-1)
    for i in outputs.sequences:
        logprobs = 0
        counter = 0
        for k in i[1:]:
            word_prob = (round(probs[0][counter][k.item()].item(), 2)) + 0.001
            logprobs = logprobs + math.log(word_prob)
            counter += 1
        out_tuple = (out_text, round(logprobs, 2))
    return out_tuple


# ask_flan_T5D is a function that takes an input text and
# returns the deterministic(do_sample=False) output of
# FLAN_T5 and logits.
def ask_flan_T5D(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt").cuda(0)
    outputs = model.generate(
        inputs,
        do_sample=False,
        eos_token_id=1,
        max_new_tokens=50,
        bos_token_id=0,
        return_dict_in_generate=True,
        output_scores=True,
    )
    out_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    probs = torch.stack(outputs.scores, dim=1).softmax(-1)
    for i in outputs.sequences:
        logprobs = 0
        counter = 0
        for k in i[1:]:
            word_prob = (round(probs[0][counter][k.item()].item(), 2)) + 0.001
            logprobs = logprobs + math.log(word_prob)
            counter += 1
        out_tuple = (out_text, round(logprobs, 2))
    return out_tuple


# Generate a topic classifier for a paragraph of text
def generate_topic(paragraph):
    results = set()
    input_text = (
        "Task: Create a topic classifier for the provided \
        paragraph.\nParagraph:\n"
        + paragraph
        + "\nTopic: "
    )
    for k in range(0, 20):
        result = ask_flan_T5(input_text)
        if result[1] > -4:
            results.add(result)
        if len(results) < 3:
            results.add(("I was wondering", -3.3))
            results.add(("I have a question", -3.3))
    sorted_results = Sort_Tuple(list(results))
    return sorted_results[0:5]


# Generate a topic classifier for a paragraph of text
def generate_topic_prefix(topic_set):
    results = set()
    for entry in topic_set:
        topic = entry[0]
        input_text = (
            "Task: Create a prepositional phrase about the topic.\n\
            Example 1\n Topic: climbing mount everest\nPrepositional \
            Phrase: With regards to climbing mount everest,\nExample \
            2\nTopic: United States Air Force\nPrepositional Phrase: \
            On the topic of the United States Air Force,\n Example 3\nTopic: "
            + topic
            + "\nPrepositional Phrase: "
        )
        for k in range(0, 5):
            results.add(ask_flan_T5(input_text))
        sorted_results = Sort_Tuple(list(results))
        return sorted_results[0:5]


# Generate who/what/where/when/why questions from a paragraph.
# Number of questions variable is an integer which indicates how
# many of each question type to try to generate.
def generate_questions(paragraph, number_of_questions):
    if len(tokenizer.encode(paragraph)) > 480:
        print("Warning, the context length is too long.")
    question_set = set()
    question_types = [
        "What",
        "Where",
        "Why",
        "How",
        "Who",
        "How much",
        "When",
        "Which"
    ]
    for qtype in question_types:
        question = (
            "Please generate a question that starts with '"
            + qtype
            + "' based on the following paragraph.\nText:\n"
            + paragraph
            + "\nQuestion:\n"
        )
        for k in range(0, number_of_questions):
            new_question = ask_flan_T5(question)
            if qtype in new_question[0]:
                question_set.add((qtype, new_question))
    return question_set


# Generate answers for a set of questions.
# Input is the paragraph of text and a set of questions where each question
# is a tuple generated from the generate_questions() function.
def generate_answers(paragraph, question_set):
    possible_answers = set()
    for question in question_set:
        input_text = (
            "Please read the following paragraph and \
            then answer the question using only data \
            found in the text. If no answer is possible, respond \
            'NA'.\nText:\n"
            + paragraph
            + "\nQuestion:\n"
            + question[1][0]
            + "\nAnswer:\n"
        )
        answer = ask_flan_T5D(input_text)
        possible_answers.add((question[0], question[1], answer))
    return possible_answers


# Generate questions from a paragraph and set of answers.
# Input is the paragraph of text and a set of answers where each question
# is a tuple generated from the generate_answers() function.
def generate_question2(paragraph, qa_set):
    qaq_results = set()
    for qa_item in qa_set:
        answer = qa_item[2][0]
        input_text = (
            "Please read the following paragraph and \
            then generate a question whose answer is: "
            + answer
            + "\nParagraph:\n"
            + paragraph
            + "\nQuestion:\n"
        )
        result = ask_flan_T5D(input_text)
        qaq_results.add((qa_item[0], qa_item[1], qa_item[2], result))
    return qaq_results


# Generate answers from a paragraph and set of questions.
# Input is the paragraph of text and a set of questions where each answer
#  is a tuple generated from the generate_questions2() function.
def generate_answers2(paragraph, question_set):
    possible_answers = set()
    for question in question_set:
        input_text = (
            "Please read the following paragraph and \
            then answer the question using only data \
            found in the text. If no answer is possible, respond \
            'NA'.\nText:\n"
            + paragraph
            + "\nQuestion:\n"
            + question
            + "\nAnswer:\n"
        )
        answer = ask_flan_T5D(input_text)
        possible_answers.add((question, answer))
    return possible_answers


# Generate declarative statement from question and answer pair.
def generate_declarative(qaq_set):
    qaqd_results = set()
    for qa_item in qaq_set:
        question = qa_item[0]
        answer = qa_item[1][0]
        if "NA" in answer:
            qaqd_results.add((question, answer,  qa_item[1]))
        else:
            input_text = (
                "Generate a declarative statement based on the \
                given question and answer pair.\nQ: What is \
                sitting on the couch?\nA: poodle\nA poodle is \
                sitting on the couch.\nQ: "
                + question
                + "\nA: "
                + answer
                + "\n"
            )
            result = ask_flan_T5D(input_text)
            qaqd_results.add((question, answer, result))
    return qaqd_results


# Generate closed book answer to question.
def generate_closed_answer(qaqd_set, topic_prefix):
    if topic_prefix:
      topic_prefix= [a[0] for a in topic_prefix]
      topic_prefix.sort(key=lambda a: len(a[0]), reverse=True)
      topic_prefix = topic_prefix[0]
    else:
      topic_prefix = None
    qaqd_results = set()
    for qa_item in qaqd_set:
        question = qa_item[0]
        answer = qa_item[2][0]
        if "NA" in answer:
            # print(answer)
            if len(qa_item) == 3:
              qaqd_results.add((qa_item[0], qa_item[1], qa_item[2], qa_item[2]))
            else:
              qaqd_results.add((qa_item[0], qa_item[1], qa_item[2], qa_item[2],  qa_item[2]))
            pass
        else:
            input_text = (
                "Task: Answer the question in a detailed fashion. \
                If the question cannot be answered without more \
                information, please answer NA.\nExample 1:\nQuestion: \
                Why does Shala like cookies?\nAnswer: It is not possible \
                to know why Shala likes cookies without more information, \
                but many people that like cookies enjoy their taste or \
                some of their ingredients (e.g. chocolate chips or \
                peanut butter).\nExample 2:\nQuestion: Why would someone \
                vote in an election?\nAnswer: There are many reasons \
                someone might vote in an election, for instance to have \
                their voice heard or to help a candidate they like win the \
                race.\nExample 3\nQuestion: What decoration goes on top of \
                a Christmas tree?\nAnswer: Usually a star is placed at the \
                top of a Christmas tree.\nExample 4:\nQuestion: "
                + (question if topic_prefix is None else (topic_prefix + " " + question))
                + "\nAnswer: "
            )
            result = ask_flan_T5D(input_text)
            if len(qa_item) == 3:
              qaqd_results.add((qa_item[0], qa_item[1], qa_item[2], result))
            else:
              qaqd_results.add((qa_item[0], qa_item[1], qa_item[2], qa_item[3], result))
    return qaqd_results

def mean_pooling(model_output, attention_mask):
    with torch.no_grad():
      token_embeddings = model_output.last_hidden_state
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
      return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

from torch.nn.functional import cosine_similarity

tokenizer, model, minilm_tokenizer, minilm_model =  None, None, None, None

def gen_qg_qa(paragraphs):
  global tokenizer, model, minilm_tokenizer, minilm_model
  # Load the model in bfloat16. Make sure to use bfloat16
  # if you are doing inference with 16bit precision.
  try:
    if tokenizer is  None: assert False
      
  except:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained(
      "google/flan-t5-large",
      device_map=device_map_T5_13B,
      torch_dtype=torch.bfloat16,
      load_in_8bit=False,
  )
    
    minilm_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    minilm_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').half().eval().cuda()  

  # Load strings as knowledge sources for QA generation.
  # You can do this with a pickle.
  #objects = []
  #with (open("paragraphs.pkl", "rb")) as openfile:
  #    while True:
  #        try:
  #            objects.append(pickle.load(openfile))
  #        except EOFError:
  #            break
  # Make sure no paragraphs are too long for T5.
  # It handles up to 512 tokens context length.
  fixed_paragraphs = []
  for k in paragraphs:
      if len(k) > 1100:
          k = k[:1100]
  
      fixed_paragraphs.append(k)
  print("Original number of paragraphs:", len(paragraphs))
  print("Length filtered number of paragraphs:", len(fixed_paragraphs))
  paragraphs = fixed_paragraphs
  
  
  # Create a dictionary of questions and answers from a list of paragraphs.
  # Takes about 20 seconds per paragraph to process.
  start_time = time.perf_counter()
  questions_dict = {}
  uniq_id = 100000
  for paragraph in paragraphs:
      topic_list = generate_topic(paragraph)
      topic_prefix = generate_topic_prefix(topic_list)
      question_set = generate_questions(paragraph, 2)
      qa_set = generate_answers(paragraph, question_set)
      qaq_set = generate_question2(paragraph, qa_set)
      q2_set = set()
      for q in qaq_set:
          q2_set.add(q[3][0])
      q2a2_set = generate_answers2(paragraph, q2_set)
      a2d_set = generate_declarative(q2a2_set)
      a3cb_set = generate_closed_answer(a2d_set, None)
      a3cb_set = generate_closed_answer(a3cb_set, topic_prefix)
      questions_dict[uniq_id] = {}
      questions_dict[uniq_id]["topics"] = topic_list
      questions_dict[uniq_id]["topic prepositions"] = topic_prefix
      questions_dict[uniq_id]["paragraph"] = paragraph
      entry_count = 0
      entry_dict = {}
      for entry in a3cb_set:
          entry_dict[entry_count] = {}
          entry_dict[entry_count]["question"] = entry[0]
          entry_dict[entry_count]["answer_T5_ob"] = entry[2][0]
          entry_dict[entry_count]["answer_T5_cb"] = entry[3][0]
          entry_dict[entry_count]["answer_T5_cb_with_prefix"] = entry[4][0]
          if entry_dict[entry_count]["answer_T5_ob"] == "NA":
            entry_dict[entry_count]["answer_T5_answer"] = "Either I do not undersand this question, or this question cannot be answered."
          else:
            toks = minilm_tokenizer(entry_dict[entry_count]["answer_T5_ob"], padding=True, truncation=True, return_tensors="pt").to('cuda')
            dat = minilm_model(**toks)
            dat = mean_pooling(dat, toks.attention_mask)
            cb_answer = entry_dict[entry_count]["answer_T5_cb"]
              
            toks = minilm_tokenizer(cb_answer, padding=True, truncation=True, return_tensors="pt").to('cuda')
            dat2 = minilm_model(**toks)
            dat2 = mean_pooling(dat2, toks.attention_mask)
            score = cosine_similarity(dat, dat2).item()
            if score < 0.75:
              entry_dict[entry_count]["answer_T5_answer"] = "I don't know. I cannot tell you the answer with the information I have."
            elif score < 0.8:
              if entry_dict[entry_count]["answer_T5_ob"].split()[0].lower() in {'the', 'this', 'a', 'an'}:
                entry_dict[entry_count]["answer_T5_answer"] = "I don't know for certain, but maybe "+entry_dict[entry_count]["answer_T5_ob"][0].lower()+ entry_dict[entry_count]["answer_T5_ob"][1:]          
              else:
                entry_dict[entry_count]["answer_T5_answer"] = "I don't know for certain, but maybe "+ entry_dict[entry_count]["answer_T5_ob"]
            elif score < 0.9:
              if entry_dict[entry_count]["answer_T5_ob"].split()[0].lower() in {'the', 'this', 'a', 'an'}:
                entry_dict[entry_count]["answer_T5_answer"] = "I believe "+ entry_dict[entry_count]["answer_T5_ob"][0].lower()+ entry_dict[entry_count]["answer_T5_ob"][1:]          
              else:
                entry_dict[entry_count]["answer_T5_answer"] = "I believe "+ entry_dict[entry_count]["answer_T5_ob"]
            else:
              entry_dict[entry_count]["answer_T5_answer"] = entry_dict[entry_count]["answer_T5_ob"] 
            entry_dict[entry_count]["answer_T5_answer_with_prefix"]  = entry_dict[entry_count]["answer_T5_answer"] 
            if len(cb_answer) < len(entry_dict[entry_count]["answer_T5_cb_with_prefix"]):
              cb_answer = entry_dict[entry_count]["answer_T5_cb_with_prefix"]

              toks = minilm_tokenizer(cb_answer, padding=True, truncation=True, return_tensors="pt").to('cuda')
              dat2 = minilm_model(**toks)
              dat2 = mean_pooling(dat2, toks.attention_mask)
              if score < cosine_similarity(dat, dat2).item():
                if cosine_similarity(dat, dat2).item() < 0.75:
                  entry_dict[entry_count]["answer_T5_answer_with_prefix"] = "I don't know. I cannot tell you the answer with the information I have."
                elif cosine_similarity(dat, dat2).item() < 0.8:
                  if entry_dict[entry_count]["answer_T5_ob"].split()[0].lower() in {'the', 'this', 'a', 'an'}:
                    entry_dict[entry_count]["answer_T5_answer_with_prefix"] = "I don't know for certain, but maybe "+entry_dict[entry_count]["answer_T5_ob"][0].lower()+ entry_dict[entry_count]["answer_T5_ob"][1:]          
                  else:
                    entry_dict[entry_count]["answer_T5_answer_with_prefix"] = "I don't know for certain, but maybe "+ entry_dict[entry_count]["answer_T5_ob"]
                elif cosine_similarity(dat, dat2).item() < 0.9:
                  if entry_dict[entry_count]["answer_T5_ob"].split()[0].lower() in {'the', 'this', 'a', 'an'}:
                    entry_dict[entry_count]["answer_T5_answer_with_prefix"] = "I believe "+ entry_dict[entry_count]["answer_T5_ob"][0].lower()+ entry_dict[entry_count]["answer_T5_ob"][1:]          
                  else:
                    entry_dict[entry_count]["answer_T5_answer_with_prefix"] = "I believe "+ entry_dict[entry_count]["answer_T5_ob"]
                else:
                  entry_dict[entry_count]["answer_T5_answer_with_prefix"] = entry_dict[entry_count]["answer_T5_ob"] 

              #'**', entry_dict[entry_count]["answer_T5_ob"], '**', entry_dict[entry_count]["answer_T5_cb"])
          entry_count += 1
      questions_dict[uniq_id]["QA_set"] = entry_dict
      uniq_id += 1
      print(uniq_id, "topics:", topic_prefix)

  stop_time = time.perf_counter()
  generation_time = stop_time - start_time
  print(questions_dict[uniq_id - 1])
  print(generation_time)

  for qd in questions_dict.values():  
      output.write(json.dumps(qd)+"\n")
  return questions_dict

