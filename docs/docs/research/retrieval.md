# Retrieval Directions and Research Papers

## 1. Retrieval-Index

At first, either a rule-based search a fixed encoder for sematic vector-based
retrieval (e.g. BERT, Contriever) could be used. The BEIR paper (see later
summary) shows still a strong performance of "rule-based" BM25, especially for
OOD data.

### Relevant Papers

1. FAISS: [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734) -
   vector index by Facebook
2. SCaNN: [https://arxiv.org/abs/1908.10396](https://arxiv.org/abs/1908.10396) -
   vector index by Google
3. BEIR:
   [https://arxiv.org/abs/2104.08663v4](https://arxiv.org/abs/2104.08663v4) -
   Benchmark for Information Retrieval
4. MS MARCO
   [https://arxiv.org/abs/1611.09268v3](https://arxiv.org/abs/1611.09268v3) -
   Machine Reading Comprehension Dataset / Retrieval Benchmark

### Links

- ElasticSearch:
  [https://www.elastic.co/elasticsearch](https://www.elastic.co/elasticsearch)
- Apache Lucene: [https://lucene.apache.org/](https://lucene.apache.org/)
- Meta Faiss:
  [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- Google Scann:
  [https://github.com/google-research/google-research/tree/master/scann](https://github.com/google-research/google-research/tree/master/scann)
- Qdrant Vector DB:
  [https://github.com/qdrant/qdrant](https://github.com/qdrant/qdrant)
- Milvus Vector DB: [https://milvus.io/](https://milvus.io/)
- Open Retrieval Index Code:
  [https://github.com/kenhktsui/open-information-retrieval](https://github.com/kenhktsui/open-information-retrieval)

## 2. Plugin-based approach

In this approach, the retrieval is used on top of a language model. It acts as
an additional tool, like a search engine for a human.

### Links

- LangChain:
  [https://github.com/hwchase17/langchain](https://github.com/hwchase17/langchain) -
  Plugins around any language model
- LlamaIndex:
  [https://github.com/jerryjliu/llama_index](https://github.com/jerryjliu/llama_index) -
  General Retrieval System for LMs and external data
- LlamaHub: [https://llamahub.ai/](https://llamahub.ai/) - Data Source Plugins
  for LlamaIndex

### Relevant Papers

- Toolformer: [http://arxiv.org/abs/2302.04761](http://arxiv.org/abs/2302.04761)
- ...

## 3. Embedding-based approach

The embedding-based approach ingests retrieved information directly into the
model, e.g. via an additional encoder and cross-attention.

### 3a

Simply inject embeddings via cross-attention or a similar mechanism.

### Relevant papers

- RETRO: [http://arxiv.org/abs/2112.04426](http://arxiv.org/abs/2112.04426)
- ...

### 3b

Inject embeddings based on a more sophisticated architecture, e.g. make the
model decide to do retrieval and only then inject embeddings. Might be hard to
train.

### 3c

Train retrieval index jointly with the injection. Possibly infeasible as the
index needs to be re-updated during training.

## Paper summaries

### Borgeaud et al 2020.: Improving Language Models by Retrieving from Trillions of Tokens - "RETRO"

Idea: Use BERT (Devlin et al. 2018) as a contextual encoder for chunks of size
64 of the training data. Then train an encoder-decoder transformer model with
inputs and similar (not too similar / same) input chunks retrieved by BERT
embedding similarity - all done in a causal way (retrieve only "from the past").
The Cross-Attention is replaced by a Chunked Cross Attention optimized for
batches of similar retrieved chunks. They pre-filter their dataset such that
data duplicates cannot easily leak information via retrieval. This was scaled to
2T tokens and a 7.5 B parameter model exceeding GPT-3 performance. RETROfitting
of a pre-trained transformer also works, with small losses in perplexity (0.3),
but a lot faster training (6 % of training sequences = 6M seq à 2048 tokens).
This is not fine-tuning but just training the cross-attention, keeping
pre-trained weights fixed. Larger models benefit from more nearest neighbors,
i.e. the 7B can utilize 40 nearest neighbor chunks, a 172M model only 10 NNs.

[http://arxiv.org/abs/2112.04426](http://arxiv.org/abs/2112.04426)

### Schick et al. 2023: Toolformer: Language Models Can Teach Themselves to Use Tools

They use in-context learning of GPT-3 and some handcrafted samples to annotate a
language modeling dataset with potential uses of external tools, like QA,
wikipedia search, a calculator, machine translation and a calendar - via text
tags for those tools and respective tool queries. They use this data then to
fine-tune GPT-2/GPT-J models, implement according tools and train with up to 25k
examples per API, max sequence length 1,024. They outperform other language
models with large margin when using tools and are comparable to larger ones when
only fine-tuned on the tool-based dataset.

[http://arxiv.org/abs/2302.04761](http://arxiv.org/abs/2302.04761)

### Guu et al 2020: REALM: Retrieval-Augmented Language Model Pre-Training

They use retrieved information from a KB to train a MLM self-supervised and
evaluate on QA tasks. Predecessor to RETRO.

The authors of the paper structure the retriever in REALM such that the
computation performed for each document can be cached and asynchronously
updated, and selection of the best documents can be formulated as Maximum Inner
Product Search (MIPS). This allows for efficient retrieval of potentially
relevant documents from a large corpus during pre-training.

During pre-training, REALM backpropagates through the retrieval step that
considers millions of documents, but it does not backpropagate to each
individual document. Instead, it uses a single encoder to encode the subset of
retrieved samples and then backpropagates through this encoder. This approach
allows for efficient computation during pre-training while still allowing for
effective utilization of world knowledge.

(https://arxiv.org/abs/2002.08909)[https://arxiv.org/abs/2002.08909]

### Zamani et al. 2022: Retrieval-Enhanced Machine Learning

(https://arxiv.org/abs/2205.01230)[https://arxiv.org/abs/2205.01230]

### Thakur et al. 2021: BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models

The BEIR benchmarking tool is designed to provide a comprehensive evaluation of
information retrieval models across diverse tasks and domains. It includes 18
retrieval datasets for comparison and evaluation of model generalization,
spanning nine different retrieval tasks such as fact checking, citation
prediction, duplicate question retrieval, argument retrieval, news retrieval,
question answering, tweet retrieval, bio-medical IR, and entity retrieval. The
selection methodology is motivated by the need for diverse tasks and domains to
evaluate the zero-shot capabilities of retrieval systems. The tool is
open-sourced with a standardized data format and easy-to-adapt code examples for
many different retrieval strategies.

They compare neural retrieval to legacy systems like BM25 and show that BM25 is
still a very strong baseline. The best model is a BM25 based search with
additional re-ranking based on a neural classifier.

Observations:

1. "In-domain performance is not a good indicator for out-of-domain
   generalization"
2. "Term-weighting fails, document expansion captures out-of-domain keyword
   vocabulary"
3. "Dense retrieval models with issues for out-of-distribution data"
4. "Re-ranking and Late-Interaction models generalize well to
   out-of-distribution data"
5. "Strong training losses for dense retrieval leads to better
   out-of-distribution performances"
6. "TAS-B model prefers to retrieve documents with shorter lengths"

Conclusion: Maybe not only focus on a vector-based index, use a standard index
as base + neural re-ranking

(https://arxiv.org/pdf/2104.08663.pdf)[https://arxiv.org/pdf/2104.08663.pdf]

## Other interesting papers

- Nakano et al: WebGPT (predecessor to ChatGPT) - fine-tune GPT3 to search the
  web for QA tasks
  (https://arxiv.org/pdf/2112.09332.pdf)[https://arxiv.org/pdf/2112.09332.pdf]

- Schick et al: PEER: A Collaborative Language Model
  (https://arxiv.org/pdf/2208.11663.pdf)[https://arxiv.org/pdf/2208.11663.pdf]

- Goyal et al. 2023: Retrieval Augmented Reinforcement Learning

- Humphreys et al. 2022: Large-Scale Retrieval for Reinforcement Learning
