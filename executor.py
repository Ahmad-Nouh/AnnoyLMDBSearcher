import copy
from typing import Dict
from jina import Executor, DocumentArray, requests
from jinahub.indexers.searcher.AnnoySearcher.annoy_searcher import AnnoySearcher
from jinahub.indexers.storage.LMDBStorage import LMDBStorage


class AnnoyLMDBSearcher(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._storage = LMDBStorage(**kwargs)
        self._vector_searcher = AnnoySearcher(**kwargs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict = None, **kwargs):
        self._vector_searcher.search(docs, parameters, **kwargs)

        kv_parameters = copy.deepcopy(parameters)

        kv_parameters["traversal_paths"] = [
            path + "m" for path in kv_parameters.get("traversal_paths", ["r"])
        ]

        self._storage.search(docs, parameters=kv_parameters, **kwargs)
