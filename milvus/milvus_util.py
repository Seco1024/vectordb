import time
import numpy as np
import json
from pymilvus import Collection, utility
import threading

def release_collection(collection_name):
    collection = Collection(name=collection_name, using='default', schema=None)
    collection.drop()


def load_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    print(dim)
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def load_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    fv = fv[:, 1:]
    return fv

def load_bvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.uint8)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 4 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 4:]
    if c_contiguous:
        fv = fv.copy()
    return fv.astype(np.float32)


def load_npy(filename):
    return np.load(filename)


def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [json.loads(line)["query"] for line in file]
    return np.array(data).astype('float32')  # Convert to NumPy array



def write_to_file(data, filename="./report/output.txt"):
    with open(filename, 'a') as f:
        f.write(data + '\n')


def insert_data_into_milvus(collection, vectors, batch_size=10000):
    num_vectors = len(vectors)
    for i in range(0, num_vectors, batch_size):
        batch_vectors = vectors[i:i + batch_size]
        batch_ids = list(range(i, i + len(batch_vectors))) 
        print(f"Inserting batch {i // batch_size + 1} of {num_vectors // batch_size}...")
        mr = collection.insert([batch_ids, batch_vectors])

    collection.flush()
    print("Data insertion completed and flushed.")

def create_ivfpq_index(collection, nlist, m):
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_PQ",
        "params": {"nlist": nlist, "m": m}
    }

    print("Creating IVFPQ index...")
    collection.create_index(field_name="vector", index_params=index_params)
    print("IVFPQ Index created.")
    
def create_hnsw_index(collection, efConstruction, M):
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": M, "efConstruction": efConstruction}
    }

    print("Creating HNSW index...")
    collection.create_index(field_name="vector", index_params=index_params)
    print("HNSW Index created.")
    
def create_diskann_index(collection):
    index_params = {
        "metric_type": "L2",
        "index_type": "DISKANN",
    }

    print("Creating DiskANN index...")
    collection.create_index(field_name="vector", index_params=index_params)
    print("DiskANN Index created.")


def measure_qps(collection, query_data, index, search_param, top_k=10, batch_size=10000):   
    print(index)
    collection.load()
    num_queries = len(query_data)
    print(num_queries)
    if index == "hnsw":
        input_params = {"params": {"ef": search_param[index]}}
        print(input_params)
    elif index == "diskann":
        input_params = {"params": {"search_list": search_param[index]}}
    else:
        input_params = {"params": {"nprobe": search_param[index]}}

    duration_list = []
    for i in range(0, num_queries, batch_size):
        query_batch = query_data[i:i + batch_size]
        print(f"Querying batch {i // batch_size + 1} of {num_queries // batch_size}...")
        start_time = time.time()
        results = collection.search(
            query_batch, "vector", param=input_params, limit=top_k
        )
        total_time = time.time() - start_time
        duration_list.append(total_time)

    overall_qps = num_queries / sum(duration_list)
    write_to_file(f"Overall QPS (Queries per second): {overall_qps:.5f}")
    write_to_file(f"Total time for {num_queries} queries: {sum(duration_list):.5f} seconds")

    return results

def compute_recall(groundtruth, predicted, k=10):
    recalls = []
    for true_neighbors, pred_neighbors in zip(groundtruth, predicted):
        true_set = set(true_neighbors[:k])
        pred_set = set(pred_neighbors[:k])
        recall = len(true_set.intersection(pred_set)) / len(true_set)
        recalls.append(recall)
    return np.mean(recalls) 