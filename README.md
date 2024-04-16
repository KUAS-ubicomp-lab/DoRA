# <B> DoRA </B>
Dual-encoder demonstration Retrievers Architecture (DoRA) is a model to retrieve in-context demonstrations for out-of-domain (OOD) tasks.

# Our Approach
We propose a novel fine-tuned demonstration retrievers architecture inspired by the bi-encoder-based dense retriever method. This consists of two encoders:
- DepRoBERTa (deproberta-large-depression) as the initial retriever to limit the pool size of the available candidates for mining the right demonstrations.
- WSW (Who-Says-What) as the scoring pre-trained language model (PLM) to score each candidate demonstration by perfectly reflecting its preferences using semantic similarity.

 # System Design 

![IJCNN 2024 Design - Enhanced Version drawio](https://github.com/KUAS-ubicomp-lab/DoRA/assets/4902204/1ad029fd-af14-463a-8c61-e209c773638e)
