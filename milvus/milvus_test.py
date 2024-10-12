import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from milvus_util import release_collection, load_fvecs, create_hnsw_index, measure_qps, load_npy, load_jsonl, insert_data_into_milvus, load_bvecs, load_ivecs, create_diskann_index, compute_recall, write_to_file, create_ivfpq_index
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
sys.path.append('../')
from util.system_monitoring import start_monitoring, stop_monitoring

os.environ["GRPC_POLL_STRATEGY"] = "poll"

dimMap = {
    "openai": 1536,
    "sift": 128,
    "gist": 960,
    "bigann": 128
}

def memmap_generator(data, chunk_size):
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:i + chunk_size][:, 4:].astype(np.float32) 


def main():
    dataset_name = sys.argv[1]
    dataset_size = sys.argv[2]
    ef_search = int(sys.argv[3])
    chunk_size = 1000000

    try:
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")
        print("Successfully connected to Milvus.")
        
        # Set schema and create collection
        collection_name = f"{dataset_name}{dataset_size}_collection"
        fields = [
            FieldSchema(name="idx", dtype=DataType.INT64, is_primary=True, auto_id=False),  # 設定 auto_id=False
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimMap[dataset_name])
        ]
        schema = CollectionSchema(
            fields=fields, description=f"{dataset_name}{dataset_size} collection"
        )
        
        # Release Collection
        # if utility.has_collection(collection_name):
        #     release_collection(collection_name)
        #     print("Collection released.")
            
        # Get Collection
        collection = Collection(name=collection_name, using="default", schema=schema)
        print("Collection get.")

        # Load Raw data
        # if dataset_name == "sift" or dataset_name == "gist":
        #     raw_data = load_fvecs(
        #         f"../data/{dataset_name}{dataset_size}/{dataset_name}_base.fvecs")
        # elif dataset_name == "openai":
        #     raw_data = load_npy(
        #         f"../data/{dataset_name}{dataset_size}/{dataset_name}_base.npy")
        # elif dataset_name == "bigann":
        #     data = np.memmap(f"../data/{dataset_name}{dataset_size}/{dataset_name}_base.bvecs",
        #                      dtype=np.uint8, mode='r', shape=(1000000000, 132))
        #     raw_data = memmap_generator(data, chunk_size)
        
        # Parallelized Data insertion 
        # monitor_thread = start_monitoring()
        # with ThreadPoolExecutor(max_workers=24) as executor:
        #     futures = []
        #     for chunk in raw_data:
        #         if len(futures) >= 24:
        #             for future in as_completed(futures):
        #                 futures.remove(future) 
        #                 break 

        #         future = executor.submit(insert_data_into_milvus, collection, chunk)
        #         futures.append(future)

        #     for future in as_completed(futures):
        #         try:
        #             future.result()
        #         except Exception as e:
        #             print(f"Error inserting data chunk: {e}")
        
        # executor.shutdown(wait=True)
        # stop_monitoring(monitor_thread, fig_name="data_insertion")
        
        # Normal Data insertion
        # monitor_thread = start_monitoring()
        # insert_data_into_milvus(collection, raw_data)
        # stop_monitoring(monitor_thread, fig_name="data_insertion")
        
        # collection.compact()
        # print(collection.get_compaction_state())
    
        # Create HNSW index
        monitor_thread = start_monitoring()
        create_hnsw_index(collection)
        stop_monitoring(monitor_thread, fig_name="index construction")

        # Create DISKANN index
        # monitor_thread = start_monitoring()
        # create_diskann_index(collection)
        # stop_monitoring(monitor_thread, fig_name="index construction(DISKANN)")
        
        # Compact
        collection.release()
        monitor_thread = start_monitoring()
        collection.compact()
        collection.wait_for_compaction_completed()
        stop_monitoring(monitor_thread, fig_name="compact")
        
        
        # Load query data
        if dataset_name == "sift" or dataset_name == "gist":
            query_data = load_fvecs(
                f"../data/{dataset_name}{dataset_size}/{dataset_name}_query.fvecs")
        elif dataset_name == "openai":
            query_data = load_jsonl(
                f"../data/{dataset_name}{dataset_size}/{dataset_name}_query.jsonl")
        elif dataset_name == "bigann":
            query_data = load_bvecs(
                f"../data/{dataset_name}{dataset_size}/{dataset_name}_query.bvecs")
        
        monitor_thread = start_monitoring()
        results = measure_qps(collection, query_data, ef_search)
        stop_monitoring(monitor_thread, fig_name="query")
        
        # calculate recall
        parsed_results = []
        for query_results in results:
            query_ids = []
            for result in query_results:
                query_ids.append(result.id)  
            parsed_results.append(query_ids)

        parsed_results = np.array(parsed_results) 
        ground_truth = np.array(load_ivecs(f"../data/{dataset_name}{dataset_size}/{dataset_name}_groundtruth.ivecs"))
        
        average_recall = compute_recall(ground_truth, parsed_results, k=10)
        write_to_file(f'Average Recall: {average_recall}\n\n')

    except Exception as e:
        print(f"An error occurred during the process: {str(e)}")
        raise e

    finally:
        connections.disconnect("default")
        print("Connection to Milvus closed.")


if __name__ == "__main__":
    main()
