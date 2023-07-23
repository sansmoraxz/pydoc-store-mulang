from docarray import DocumentArray, Document
import numpy as np
from jina import Executor, requests, Deployment, dynamic_batching
import torch


class MyExec(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=128,
                out_features=128,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=32)
        )
        
    @requests(on='/foo')
    @dynamic_batching(preferred_batch_size=10, timeout=200)
    def foo(self, docs: DocumentArray, **kwargs):
        return docs

with Deployment(uses=MyExec, port=12345, replicas=2) as dep:
    arr = np.random.random((500, 128))
    arr = DocumentArray.from_ndarray(arr)
    docs = dep.post(on='/foo', inputs=arr, on_done=print)
    print("Done")
    #print(docs.texts)
