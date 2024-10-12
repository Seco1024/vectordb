import sys
from pymilvus import connections, utility, Collection, list_collections, drop_collection, client
from milvus_util import release_collection, load_npy, load_jsonl, write_to_file, load_ivecs, load_fvecs
import numpy as np
import tensorflow_datasets as tfds


def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Successfully connected to Milvus")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise e


def get_collection_info(collection_name):
    try:
        if not utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return None

        collection = Collection(name=collection_name)

        num_entities = collection.num_entities
        print(
            f"Collection '{collection_name}' contains {num_entities} entities.")

        vector_size = 128
        bytes_per_vector = vector_size
        total_bytes = num_entities * bytes_per_vector
        total_megabytes = total_bytes / (1024 ** 2)
        print(f"Approximate collection size: {total_megabytes:.2f} MB")

        indexes = collection.indexes
        if len(indexes) > 0:
            for index in indexes:
                print(f"Index {index.index_name}: {index.params}")
        else:
            print(f"No index found for collection '{collection_name}'.")

        return collection

    except Exception as e:
        print(f"An error occurred: {e}")


def release_collections():
    connect_to_milvus()
    collections = list_collections()
    for collection in collections:
        drop_collection(collection)
        print(f"Deleted collection: {collection}")

def release_collection(collection_name):        
    connect_to_milvus()
    collection = get_collection_info(collection_name)
    if collection:
        print(f"releasing collection {collection_name}")
        release_collection(collection)
    connections.disconnect("default")
    

if __name__ == "__main__":
    connect_to_milvus()
    collection = Collection(name="gist1m_collection")

    # collection.compact()
    # print("compact finish")
    # collection.load()
    # get_collection_info("bigann1b_collection")
    # print(utility.list_collections())

    # collection = Collection(name="sift1m_collection")
    # collection.release()
    # collection.drop()
    
    # raw_data = load_ivecs(f"../data/sift1m/sift_groundtruth.ivecs")
    # print(raw_data[12])
    
    # Check active connections
    # active_connections = connections.list_connections()
    # print("Active connections:", active_connections)
    
    # Release Collection
    # collection = Collection("bigann1b_collection")     
    collection.release()
    
    # collection = Collection("bigann1b_collection")
    collection.drop_index()