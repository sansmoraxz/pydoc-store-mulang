from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

transformer = SentenceTransformer('all-MiniLM-L6-v2')


def create_milvus_collection(collection_name, title_dim, contents_dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="title_vector", dtype=DataType.FLOAT_VECTOR, dim=title_dim),
        FieldSchema(name="contents", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="contents_vector", dtype=DataType.FLOAT_VECTOR, dim=contents_dim),
        FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="tags_vector", dtype=DataType.FLOAT_VECTOR, dim=8)
    ]

    schema = CollectionSchema(fields=fields, description='search text')
    title_idx_params = {
        'metric_type': "L2",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    contents_idx_params = {
        'metric_type': "L2",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    tags_idx_params = {
        'metric_type': "L2",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection = Collection(name=collection_name, schema=schema)

    collection.create_index(field_name='title_vector', index_params=title_idx_params)
    collection.create_index(field_name='contents_vector', index_params=contents_idx_params)
    collection.create_index(field_name='tags_vector', index_params=tags_idx_params)

    return collection


def create_milvus_connection(host, port):
    connections.connect(host=host, port=port)


def search_milvus_collection(collection_name, query_records, top_k):
    collection = Collection(name=collection_name)
    results = collection.search(data=query_records, anns_field="title_vector", param={"nprobe": 16}, limit=top_k)
    return results


def embed_insert(collection, id, title, contents, tags):
    titles_vec = transformer.encode(title)
    contents_vec = transformer.encode(contents)
    tags_vec = transformer.encode(tags)
    collection.insert([
        [id, title, titles_vec, contents, contents_vec, tags, tags_vec]
    ])
