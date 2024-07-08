# <B> <I> DoRA</I>: <u>D</u>ual-encoder dem<u>o</u>nstration <u>R</u>etriever <u>A</u>rchitecture to Transfer Large Language Models for Depression Detection </B>
This repository contains the source code of the initial version of <I>DoRA</I>, a novel dual-encoder retriever architecture designed to identify the most appropriate demonstrations by pairing them with the soft embeddings of input source prompt sequences. Our objective is to adapt a (large language model's) LLM’s knowledge of semantic modelling in multi-party conversations (MPCs) to detect depressive content using out-of-domain (OOD) transferability of Instruction Prompt Tuning (IPT). To our knowledge, this is the first attempt to use OOD transferability of IPT for depression detection in text-based MPCs. Multiple prompt templates and soft verbalizers are specially crafted to support the core logic of the prompt manager for depressed utterance classification (DUC) and depressed speaker identification (DSI). 

# Datasets
- Download and extract the [Reddit SDD Corpus](https://ir.cs.georgetown.edu/resources/rsdd.html)
- Download and use the [Reddit eRisk 18 T2 2018](https://link.springer.com/chapter/10.1007/978-3-319-98932-7_30)
- Download and use the [Reddit eRisk 22 T2 2022](https://books.google.co.jp/books?hl=en&lr=&id=LzaFEAAAQBAJ&oi=fnd&pg=PA231&dq=Overview+of+eRisk+2022:+Early+Risk+Prediction+on+the+Internet&ots=LnO4GFgjt7&sig=lgSXnAWqqgjiPUp-jYV3HKIv4z8&redir_esc=y#v=onepage&q=Overview%20of%20eRisk%202022%3A%20Early%20Risk%20Prediction%20on%20the%20Internet&f=false)
- Download and extract the [Twitter Depression 2022](https://www.nature.com/articles/s41599-022-01313-2)

# <I>DoRA</I> Training
Demonstrations scoring and ranking determines the top-k demonstrations using the supervisory signals of the frozen LLM, focusing on MPC structure and semantic modelling relevant to the unseen task. Reddit SDD corpus is used for demonstrations scoring and ranking purposes. Top-ranked demonstrations for each utterance in the MPC prompt are subsequently used for classification tasks in the prompt manager. 
```
python demonstrations_finder.py

python demonstrations_scorer.py
```
Demonstrations training involves optimizing the dual-encoder retriever based on LLM signals in MPC modelling and candidate demonstrations for the unseen task, without ground truths. 
```
python retriever_trainer.py
```
Prompt manager integrates the components in-context demonstrations, prompt template, and prompt verbalizer to classify each utterance in the MPC prompt as depressive or normal using the OpenPrompt framework.
```
python prompt_model/train.py
```

# Settings
Python 3.8 and PyTorch 2.0 were used as the main programming language and machine learning framework, respectively . We separated MPC data into three categories based on the session length such as Len-5, Len-10, and Len-15 and used two different prompt lengths <I> (l) </I> such as 50, and 75. Hyper-parameters were used such as GELU activations, Adam optimizer, with learning rate 0.0005, warmup proportion 0.1, and frozen model hyper-parameters, θ1 and θ2 both True. Number of candidate demonstrations and ranked demonstrations were kept as 100 and 5, respectively.

# Baseline Models
To evaluate the performance of <I>DoRA</I> against previous state-of-the-art (SOTA) methods for demonstration retrieval, we used the following methods.
- BM25: a term-based scoring method.
- SBERT: a dense retriever for obtaining semantically meaningful sentence embeddings.
- KATE: an off-the-shelf retriever for identifying semantically similar demonstrations for GPT-3.
- EPR: a retriever based on LLM signals for in-domain tasks.
- UPRISE: a universal retriever for multiple tasks.
- CEIL: an iterative method to identify diverse few-shot demonstrations.
- UDR: an iterative mining method to identify demonstration candidates for various LLM signals.

WSW, BERT, RoBERTa, SA-BERT, MPC-BERT, ELECTRA, MDFN, and DisorBERT were used as pre-trained language models (PLMs). LLaMA 2-7B and OPT-7B were used as open-source LLMs and GPT-3, ChatGPT, and GPT-4 were adopted as closed-source LLMs to evaluate both DUC and DSI.

# Metrics
