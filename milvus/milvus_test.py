import os
import sys
import numpy as np
import argparse
import requests
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from milvus_util import release_collection, load_fvecs, create_hnsw_index, measure_qps, load_npy, load_jsonl, insert_data_into_milvus, load_bvecs, load_ivecs, create_diskann_index, compute_recall, write_to_file, create_ivfpq_index
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
os.environ["GRPC_POLL_STRATEGY"] = "poll"
sys.path.append('../')
from util.system_monitoring import start_monitoring, stop_monitoring


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sift')
parser.add_argument('--size', type=str, default='1m')
parser.add_argument('--index', type=str, default="hnsw")
parser.add_argument('--release_collection', type=bool, default=False)
parser.add_argument('--insert_data', type=bool, default=False)  # phase 1
parser.add_argument('--cores', type=int, default=30)  # phase 1
parser.add_argument('--build_index', type=bool, default=False)  # phase 2
parser.add_argument('--efConstruction', type=int, default=200)  # phase 2 (hnsw)
parser.add_argument('--M', type=int, default=16)  # phase 2 (hnsw)
parser.add_argument('--nlist', type=int, default=1024)  # phase 2 (ivfpq)
parser.add_argument('--m', type=int, default=16)  # phase 2  (ivfpq)
parser.add_argument('--compaction', type=bool, default=False)  # phase 3
parser.add_argument('--loop', type=bool, default=False)  # phase 4
parser.add_argument('--k', type=int, default=10)  # phase 4
parser.add_argument('--batch', type=int, default=10000)  # phase 4
parser.add_argument('--efSearch', type=int, default=40)  # phase 4 (hnsw)
parser.add_argument('--searchList', type=int, default=40)  # phase 4 (diskann)
parser.add_argument('--nProbe', type=int, default=40)  # phase 4 (ivfpq)
args = parser.parse_args()

dimMap = {
    "openai": 1536,
    "sift": 128,
    "gist": 960,
    "bigann": 128
}


def start_profiling(output_file, profile_type="profile"):
    curl_command = [
        "curl", f"http://localhost:9091/debug/pprof/{profile_type}",
        "--output", output_file
    ]
    process = subprocess.Popen(curl_command)
    return process

def stop_profiling(process):
    process.terminate()
    process.wait()
    
def memmap_generator(data, chunk_size):
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:i + chunk_size][:, 4:].astype(np.float32) 


def main():
    try:
        connections.connect("default", host="localhost", port="19530")
        print("Successfully connected to Milvus.")
        
        # Get collection
        collection_name = f"{args.dataset}{args.size}_collection"
        fields = [
            FieldSchema(name="idx", dtype=DataType.INT64, is_primary=True, auto_id=False),  
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimMap[args.dataset])
        ]
        schema = CollectionSchema(fields=fields, description=f"{args.dataset}{args.size}_collection")
        
        # Release Collection
        if args.release_collection == True and utility.has_collection(collection_name):
            release_collection(collection_name)
            print("Collection released.")

        collection = Collection(name=collection_name, using="default", schema=schema)
        # collection = Collection(name=collection_name, using="default")
        print("Collection get.")


        # Phase 1: Load & Insert
        dataset_name = args.dataset
        dataset_size = args.size
        index = args.index 
        chunk_size = 1000000
        
        if args.insert_data:
            if dataset_name == "sift" or dataset_name == "gist":
                raw_data = load_fvecs(
                    f"../data/{dataset_name}{dataset_size}/{dataset_name}_base.fvecs")
            elif dataset_name == "openai":
                raw_data = load_npy(
                    f"../data/{dataset_name}{dataset_size}/{dataset_name}_base.npy")
            elif dataset_name == "bigann":
                data = np.memmap(f"../data/{dataset_name}{dataset_size}/{dataset_name}_base.bvecs",
                                dtype=np.uint8, mode='r', shape=(1000000000, 132))
                raw_data = memmap_generator(data, chunk_size)
        
            # Parallelized Data insertion 
            profiling_process_cpu = start_profiling("./pprof/cpu_profile_phase1.pb", "profile?seconds=500")
            monitor_thread = start_monitoring()
            with ThreadPoolExecutor(max_workers=args.cores) as executor:
                futures = []
                for chunk in raw_data:
                    if len(futures) >= args.cores:
                        for future in as_completed(futures):
                            futures.remove(future) 
                            break 

                    future = executor.submit(insert_data_into_milvus, collection, chunk)
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error inserting data chunk: {e}")
        
            executor.shutdown(wait=True)
            profiling_process_heap = start_profiling("./pprof/heap_profile_phase1.pb", "heap")
            stop_monitoring(monitor_thread, fig_name="Data Insertion")
            time.sleep(3)
            stop_profiling(profiling_process_cpu)
            stop_profiling(profiling_process_heap)
        
            # profiling_process_cpu = start_profiling("./pprof/cpu_profile_phase1.pb", "profile?seconds=15")
            # monitor_thread = start_monitoring()
            # insert_data_into_milvus(collection, raw_data)
            # stop_monitoring(monitor_thread, fig_name="data_insertion")
            # profiling_process_heap = start_profiling("./pprof/heap_profile_phase1.pb", "heap")
            # time.sleep(3)
            # stop_profiling(profiling_process_cpu)
            # stop_profiling(profiling_process_heap)


        # Phase 2: Build Index
        profiling_process_cpu = start_profiling("./pprof/cpu_profile_phase2.pb", "profile?seconds=100")

        if args.build_index:
            monitor_thread = start_monitoring()
            if index == "hnsw":
                create_hnsw_index(collection, args.efConstruction, args.M)
                profiling_process_heap = start_profiling("./pprof/heap_profile_phase2.pb", "heap")
            elif index == "ivfpq":
                create_ivfpq_index(collection, args.nlist, args.m)
            elif index == "diskann":
                create_diskann_index(collection)
            stop_monitoring(monitor_thread, fig_name="Index Building")
            
                    
        time.sleep(3)
        stop_profiling(profiling_process_cpu)
        stop_profiling(profiling_process_heap)
            
            
        # Phase 3: Compaction manually
        if args.compaction:
            profiling_process_cpu = start_profiling("./pprof/cpu_profile_phase3.pb", "profile?seconds=100")

            monitor_thread = start_monitoring()
            collection.compact()
            print("Compacting...")
            collection.wait_for_compaction_completed()
            stop_monitoring(monitor_thread, fig_name="Compaction")
            
            profiling_process_heap = start_profiling("./pprof/heap_profile_phase3.pb", "heap")
            time.sleep(3)
            stop_profiling(profiling_process_cpu)
            stop_profiling(profiling_process_heap)
        
        
        # Phase 4: Query
        params = {"hnsw": args.efSearch, "diskann": args.searchList, "ivfpq": args.nProbe}
        
        # Load data
        if dataset_name == "sift" or dataset_name == "gist":
            query_data = load_fvecs(
                f"../data/{dataset_name}{dataset_size}/{dataset_name}_query.fvecs")
            ground_truth = np.array(load_ivecs(f"../data/{dataset_name}{dataset_size}/{dataset_name}_groundtruth.ivecs"))
        elif dataset_name == "openai":
            query_data = load_jsonl(
                f"../data/{dataset_name}{dataset_size}/{dataset_name}_query.jsonl")
        elif dataset_name == "bigann":
            query_data = load_bvecs(
                f"../data/{dataset_name}{dataset_size}/{dataset_name}_query.bvecs")
            ground_truth = np.array(load_ivecs(f"../data/{dataset_name}{dataset_size}/gnd/idx_1000M.ivecs"))
        
        # Issue queries
        if args.loop:
            num_iterations = 5
        else:
            num_iterations = 1
        
        profiling_process_cpu = start_profiling("./pprof/cpu_profile_phase4.pb", "profile?seconds=1000")
        
        for i in range(0, num_iterations):
            if num_iterations == 1:
                params = {"hnsw": args.efSearch, "diskann": args.searchList, "ivfpq": args.nProbe}
            else:
                params = {"hnsw": 9 + 2**i, "diskann": 2**i, "ivfpq": 2**i}
            monitor_thread = start_monitoring()
            results = measure_qps(collection, query_data, index, params, args.k, args.batch)
            profiling_process_heap = start_profiling("./pprof/heap_profile_phase4.pb", "heap")
            stop_monitoring(monitor_thread, fig_name="Query")
            
            # Calculate recall
            parsed_results = []
            for query_results in results:
                query_ids = []
                for result in query_results:
                    query_ids.append(result.id)  
                parsed_results.append(query_ids)

            parsed_results = np.array(parsed_results) 
            average_recall = compute_recall(ground_truth, parsed_results, k=10)
            write_to_file(f'Average Recall: {average_recall}\n\n')
            
        time.sleep(3)
        stop_profiling(profiling_process_cpu)
        stop_profiling(profiling_process_heap)

    except Exception as e:
        print(f"An error occurred during the process: {str(e)}")
        stop_profiling(profiling_process_cpu)
        stop_profiling(profiling_process_heap)
        raise e

    finally:
        connections.disconnect("default")
        print("Connection to Milvus closed.")


if __name__ == "__main__":
    main()
