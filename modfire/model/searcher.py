import abc
import math

import faiss
import numpy as np
import rich


class Searcher(abc.ABC):
    @abc.abstractmethod
    def add(self, database: np.ndarray):
        raise NotImplementedError
    @abc.abstractmethod
    def search(self, query: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BinarySearcher(Searcher):
    def __init__(self, bits: int):
        self.bits = bits
        if self.bits % 8 != 0 and self.bits % 12 != 0 and self.bits % 16 != 0:
            raise ValueError(f"Please use (8*N) or (12*N) or (16*N)-bit hash codes. Current: {bits}-bit.")
        self.binary = faiss.IndexBinaryFlat(bits)
        self.index = faiss.IndexBinaryIDMap(self.binary)

    def reset(self):
        return self.index.reset()

    def remove(self, ids: np.ndarray):
        return self.index.remove_ids(ids)

    def add(self, database: np.ndarray, ids: np.ndarray):
        if database.dtype != np.uint8:
            raise ValueError("Array to be indexed must be encoded to uint8.")
        if len(database.shape) != 2 or database.shape[-1] != self.bits // 8:
            raise ValueError(f"Database shape wrong. Expect: [N, {self.bits // 8}]. Got: {[database.shape]}.")
        self.index.add_with_ids(database, ids)

    def search(self, query: np.ndarray, numReturns: int) -> np.ndarray:
        if query.dtype != np.uint8:
            raise ValueError("Query array must be encoded to uint8.")
        if len(query.shape) != 2 or query.shape[-1] != self.bits // 8:
            raise ValueError(f"Query shape wrong. Expect: [N, {self.bits // 8}]. Got: {[query.shape]}.")
        if numReturns < 0:
            numReturns = self.index.ntotal
        _, indices = self.index.search(query, numReturns)
        return indices


class PQSearcher(Searcher):
    def __init__(self, codebook: np.ndarray):
        M, K, D = codebook.shape
        if K != 256 and K != 4096 and K != 65536:
            raise ValueError(f"Please use 8,12,16-bit quantization. Current: {int(math.log2(K))}-bit.")
        self.pq = faiss.IndexPQ(D * M, M, int(math.log2(K)))
        self.index = faiss.IndexIDMap(self.pq)
        # Codebook params
        self.M = M
        self.K = K
        self.D = D * M
        self.assignCodebook(codebook)

    def assignCodebook(self, codebook: np.ndarray):
        M, K, D = codebook.shape
        if self.M != M or self.K != K or self.D != (D * M):
            raise ValueError(f"Codebook shape mis-match. Expect: {[self.M, self.K, self.D]}, Got: {[M, K, D]}.")
        faiss.copy_array_to_vector(codebook.ravel(), self.pq.pq.centroids)
        self.index.is_trained = True
        self.pq.is_trained = True

    def reset(self):
        return self.index.reset()

    def remove(self, ids: np.ndarray):
        return self.index.remove_ids(ids)

    def add(self, database: np.ndarray, ids: np.ndarray):
        if len(database.shape) != 2 or database.shape[-1] != self.D:
            raise ValueError(f"Database shape wrong. Expect: [N, {self.D}]. Got: {[database.shape]}.")
        self.index.add_with_ids(database, ids)

    def search(self, query: np.ndarray, numReturns: int) -> np.ndarray:
        if len(query.shape) != 2 or query.shape[-1] != self.D:
            raise ValueError(f"Query shape wrong. Expect: [N, {self.D}]. Got: {[query.shape]}.")
        if numReturns < 0:
            numReturns = self.index.ntotal
        _, indices = self.index.search(query, numReturns)
        return indices
