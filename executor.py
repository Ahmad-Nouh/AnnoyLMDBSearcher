import os
import copy
import lmdb
import numpy as np
from typing import Dict, List, Optional
from annoy import AnnoyIndex
from jina import Document, DocumentArray, Executor, requests
from jina_commons import get_logger
from jina_commons.indexers.dump import import_vectors, export_dump_streaming, import_metas


class _LMDBHandler:
    def __init__(self, file, map_size):
        # see https://lmdb.readthedocs.io/en/release/#environment-class for usage
        self.file = file
        self.map_size = map_size

    @property
    def env(self):
        return self._env

    def __enter__(self):
        self._env = lmdb.Environment(
            self.file,
            map_size=self.map_size,
            subdir=False,
            readonly=False,
            metasync=True,
            sync=True,
            map_async=False,
            mode=493,
            create=True,
            readahead=True,
            writemap=False,
            meminit=True,
            max_readers=126,
            max_dbs=0,  # means only one db
            max_spare_txns=1,
            lock=True,
        )
        return self._env

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, '_env'):
            self._env.close()


class LMDBStorage(Executor):
    """An lmdb-based Storage Indexer for Jina

    For more information on lmdb check their documentation: https://lmdb.readthedocs.io/en/release/
    """

    def __init__(
            self,
            map_size: int = 1048576000,  # in bytes, 1000 MB
            default_traversal_paths: List[str] = ['r'],
            dump_path: str = None,
            default_return_embeddings: bool = True,
            *args,
            **kwargs,
    ):
        """
        :param map_size: the maximal size of teh database. Check more information at
            https://lmdb.readthedocs.io/en/release/#environment-class
        :param default_traversal_paths: fallback traversal path in case there is not traversal path sent in the request
        :param default_return_embeddings: whether to return embeddings on search or not
        """
        super().__init__(*args, **kwargs)
        self.map_size = map_size
        self.default_traversal_paths = default_traversal_paths
        self.file = os.path.join(self.workspace, 'db.lmdb')
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
        self.logger = get_logger(self)

        self.dump_path = dump_path or kwargs.get('runtime_args', {}).get(
            'dump_path', None
        )
        if self.dump_path is not None:
            self.logger.info(f'Importing data from {self.dump_path}')
            ids, metas = import_metas(self.dump_path, str(self.runtime_args.pea_id))
            da = DocumentArray()
            for id, meta in zip(ids, metas):
                serialized_doc = Document(meta)
                serialized_doc.id = id
                da.append(serialized_doc)
            self.index(da, parameters={'traversal_paths': ['r']})
        self.default_return_embeddings = default_return_embeddings

    def _handler(self):
        # required to create a new connection to the same file
        # on each subprocess
        # https://github.com/jnwatson/py-lmdb/issues/289
        return _LMDBHandler(self.file, self.map_size)

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Add entries to the index

        :param docs: the documents to add
        :param parameters: parameters to the request
        """
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        if docs is None:
            return
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                for d in docs.traverse_flat(traversal_paths):
                    transaction.put(d.id.encode(), d.SerializeToString())

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Update entries from the index by id

        :param docs: the documents to update
        :param parameters: parameters to the request
        """
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        if docs is None:
            return
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                for d in docs.traverse_flat(traversal_paths):
                    # TODO figure out if there is a better way to do update in LMDB
                    # issue: the defacto update method is an upsert (if a value didn't exist, it is created)
                    # see https://lmdb.readthedocs.io/en/release/#lmdb.Cursor.replace
                    if transaction.delete(d.id.encode()):
                        transaction.replace(d.id.encode(), d.SerializeToString())

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param docs: the documents to delete
        :param parameters: parameters to the request
        """
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        if docs is None:
            return
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                for d in docs.traverse_flat(traversal_paths):
                    transaction.delete(d.id.encode())

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Retrieve Document contents by ids

        :param docs: the list of Documents (they only need to contain the ids)
        :param parameters: the parameters for this request
        """
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        return_embeddings = parameters.get(
            'return_embeddings', self.default_return_embeddings
        )
        if docs is None:
            return
        docs_to_get = docs.traverse_flat(traversal_paths)
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                for i, d in enumerate(docs_to_get):
                    id = d.id
                    serialized_doc = Document(transaction.get(d.id.encode()))
                    if not return_embeddings:
                        serialized_doc.pop('embedding')
                    d.update(serialized_doc)
                    d.id = id

    @requests(on='/dump')
    def dump(self, parameters: Dict, **kwargs):
        """Dump data from the index

        Requires
        - dump_path
        - shards
        to be part of `parameters`

        :param parameters: parameters to the request"""
        path = parameters.get('dump_path', None)
        if path is None:
            self.logger.error('parameters["dump_path"] was None')
            return

        shards = parameters.get('shards', None)
        if shards is None:
            self.logger.error('parameters["shards"] was None')
            return
        shards = int(shards)

        export_dump_streaming(path, shards, self.size, self._dump_generator())

    @property
    def size(self):
        """Compute size (nr of elements in lmdb)"""
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                stats = transaction.stat()
                return stats['entries']

    def _dump_generator(self):
        with self._handler() as env:
            with env.begin(write=True) as transaction:
                cursor = transaction.cursor()
                cursor.iternext()
                iterator = cursor.iternext(keys=True, values=True)
                for it in iterator:
                    id, data = it
                    doc = Document(data)
                    yield id.decode(), doc.embedding, LMDBStorage._doc_without_embedding(
                        doc
                    ).SerializeToString()

    @staticmethod
    def _doc_without_embedding(d):
        new_doc = Document(d, copy=True)
        new_doc.ClearField('embedding')
        return new_doc


class AnnoySearcher(Executor):
    """Annoy powered vector indexer

    For more information about the Annoy supported parameters, please consult:
        - https://github.com/spotify/annoy

    .. note::
        Annoy package dependency is only required at the query time.
    """

    def __init__(
            self,
            default_top_k: int = 10,
            metric: str = "cosine",
            num_trees: int = 10,
            dump_path: Optional[str] = None,
            default_traversal_paths: List[str] = ["r"],
            is_distance: bool = True,
            **kwargs,
    ):
        """
        Initialize an AnnoyIndexer

        :param default_top_k: get tok k vectors
        :param metric: Metric can be "euclidean", "cosine", or "inner_product"
        :param num_trees: builds a forest of n_trees trees. More trees gives higher precision when querying.
        :param dump_path: the path to load ids and vecs
        :param traverse_path: traverse path on docs, e.g. ['r'], ['c']
        :param is_distance: Boolean flag that describes if distance metric need to be reinterpreted as similarities.
        """
        super().__init__(**kwargs)
        self.default_top_k = default_top_k
        self.metric = metric
        self.num_trees = num_trees
        self.default_traversal_paths = default_traversal_paths
        self.is_distance = is_distance
        self.logger = get_logger(self)
        self._doc_id_to_offset = {}
        dump_path = dump_path or kwargs.get("runtime_args", {}).get("dump_path", None)
        if dump_path is not None:
            self.logger.info('Start building "AnnoyIndexer" from dump data')
            ids, vecs = import_vectors(dump_path, str(self.runtime_args.pea_id))
            self._ids = np.array(list(ids))
            self._vecs = np.array(list(vecs))
            num_dim = self._vecs.shape[1]
            self._indexer = AnnoyIndex(num_dim, self.metric_type)
            self._load_index(self._ids, self._vecs)
            self.logger.info("Done building Annoy index")
        else:
            self.logger.warning(
                'No data loaded in "AnnoyIndexer". Use .rolling_update() to re-initialize it...'
            )

    def _load_index(self, ids, vecs):
        for idx, v in enumerate(vecs):
            self._indexer.add_item(idx, v.astype(np.float32))
            self._doc_id_to_offset[ids[idx]] = idx
        self._indexer.build(self.num_trees)

    @requests(on="/search")
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        if not hasattr(self, "_indexer"):
            self.logger.warning("Querying against an empty index")
            return

        traversal_paths = parameters.get(
            "traversal_paths", self.default_traversal_paths
        )
        top_k = parameters.get("top_k", self.default_top_k)

        for doc in docs.traverse_flat(traversal_paths):
            indices, dists = self._indexer.get_nns_by_vector(
                doc.embedding, top_k, include_distances=True
            )
            for idx, dist in zip(indices, dists):
                match = Document(id=self._ids[idx], embedding=self._vecs[idx])
                if self.is_distance:
                    if self.metric == "inner_product":
                        match.scores[self.metric] = 1 - dist
                    else:
                        match.scores[self.metric] = dist
                else:
                    if self.metric == "inner_product":
                        match.scores[self.metric] = dist
                    elif self.metric == "cosine":
                        match.scores[self.metric] = 1 - dist
                    else:
                        match.scores[self.metric] = 1 / (1 + dist)
                doc.matches.append(match)

    @requests(on="/fill_embedding")
    def fill_embedding(self, query_da: DocumentArray, **kwargs):
        for doc in query_da:
            doc_idx = self._doc_id_to_offset.get(doc.id)
            if doc_idx is not None:
                doc.embedding = np.array(self._indexer.get_item_vector(int(doc_idx)))
            else:
                self.logger.warning(f"Document {doc.id} not found in index")

    @property
    def metric_type(self):
        metric_type = 'euclidean'
        if self.metric == 'cosine':
            metric_type = 'angular'
        elif self.metric == 'inner_product':
            metric_type = 'dot'

        if self.metric not in ['euclidean', 'cosine', 'inner_product']:
            self.logger.warning(
                f'Invalid distance metric {self.metric} for Annoy index construction! '
                'Default to euclidean distance'
            )

        return metric_type


class AnnoyLMDBSearcher(Executor):
    def __init__(self, dump_path=None, **kwargs):
        super().__init__(**kwargs)
        self._storage = LMDBStorage(dump_path, **kwargs)
        self._vector_searcher = AnnoySearcher(dump_path, **kwargs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict = None, **kwargs):
        self._vector_searcher.search(docs, parameters, **kwargs)

        kv_parameters = copy.deepcopy(parameters)

        kv_parameters["traversal_paths"] = [
            path + "m" for path in kv_parameters.get("traversal_paths", ["r"])
        ]

        self._storage.search(docs, parameters=kv_parameters, **kwargs)
