# AnnoyLMDBSearcher
AnnoyLMDBSearcher is a compound Searcher Executor for Jina, made up of [AnnoySearcher](https://hub.jina.ai/executor/wiu040h9) for performing similarity search on the embeddings, and of [LMDBStorage](https://hub.jina.ai/executor/scdc6dop) for retrieving the metadata of the Documents.

## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://AnnoyLMDBSearcher')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://AnnoyLMDBSearcher')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
