import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient,models

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from faiss_baseline import load_cached_truth,load_database_vectors,load_fvecs,load_labels,load_query_indices,load_truth,percentile,recall_at_k,select_nprobe


def directory_megabytes(path):
    total=0
    for root,_,files in os.walk(path):
        for filename in files:
            total+=os.path.getsize(os.path.join(root,filename))
    return total/1024/1024


def upload_dataset(client,collection,vectors,labels,batch_size):
    count=len(vectors)
    for start in range(0,count,batch_size):
        stop=min(count,start+batch_size)
        points=[models.PointStruct(id=index,vector=vectors[index].tolist(),payload={"label":int(labels[index])}) for index in range(start,stop)]
        client.upsert(collection_name=collection,points=points,wait=True)
        if stop%5_000<batch_size or stop==count:
            print(f"qdrant uploaded={stop}/{count}",flush=True)


def wait_for_index(client,collection,timeout_seconds):
    start=time.time()
    while True:
        info=client.get_collection(collection)
        status=str(info.status).lower()
        print(f"qdrant status={status} indexed={info.indexed_vectors_count}",flush=True)
        if "green" in status and int(info.indexed_vectors_count or 0)>=int(info.points_count or 0):
            return info
        if time.time()-start>timeout_seconds:
            raise TimeoutError("qdrant index did not finish")
        time.sleep(2)


def evaluate(client,collection,queries,query_labels,truth,indices,ef,k,repetitions,acorn,exact=False):
    recalls=[]
    times=[]
    for query_position,source_index in enumerate(indices):
        condition=models.Filter(must=[models.FieldCondition(key="label",match=models.MatchValue(value=int(query_labels[source_index])))])
        acorn_parameters=models.AcornSearchParams(enable=True,max_selectivity=1.0) if acorn else None
        parameters=models.SearchParams(hnsw_ef=ef if not exact else None,exact=exact,acorn=acorn_parameters)
        request=lambda:client.query_points(collection_name=collection,query=queries[source_index].tolist(),query_filter=condition,search_params=parameters,limit=k,with_payload=False,with_vectors=False).points
        request()
        latest=None
        for _ in range(repetitions):
            start=time.perf_counter_ns()
            latest=request()
            times.append((time.perf_counter_ns()-start)/1_000_000)
        recalls.append(recall_at_k([point.id for point in latest],truth[query_position],k))
    return {
        "recall":float(np.mean(recalls)),
        "p50_ms":percentile(times,50),
        "p95_ms":percentile(times,95),
        "qps":1000/float(np.mean(times)),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    split_group=parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--sen-environment")
    split_group.add_argument("--query-partitions")
    parser.add_argument("--output",required=True)
    parser.add_argument("--truth-cache")
    parser.add_argument("--storage",required=True)
    parser.add_argument("--collection",default="sen_arxiv_fanns_100k")
    parser.add_argument("--target-recall",type=float,default=0.95)
    parser.add_argument("--selection-margin",type=float,default=0.02)
    parser.add_argument("--safety-factor",type=float,default=2.0)
    parser.add_argument("--repetitions",type=int,default=2)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--rebuild",action="store_true")
    parser.add_argument("--acorn",action="store_true")
    parser.add_argument("--exact",action="store_true")
    parser.add_argument("--k",type=int,default=10)
    args=parser.parse_args()
    data_path=Path(args.data)
    output_path=Path(args.output)
    output_path.mkdir(parents=True,exist_ok=True)
    build_path=output_path/"qdrant_build.json"
    train_indices,heldout_indices,k=load_query_indices(args.sen_environment,args.query_partitions,args.k)
    vectors=load_database_vectors(data_path)
    queries=load_fvecs(data_path/"query_vectors.fvecs")
    labels=load_labels(data_path/"database_attributes.jsonl")
    query_labels=np.asarray([json.loads(line)["label"] for line in open(data_path/"em_query_attributes.jsonl",encoding="utf-8")],dtype=np.int64)
    if args.truth_cache:
        train_truth=load_cached_truth(args.truth_cache,train_indices,k)
        heldout_truth=load_cached_truth(args.truth_cache,heldout_indices,k)
    else:
        train_truth=load_truth(data_path/"ground_truth_em.ivecs",train_indices,k)
        heldout_truth=load_truth(data_path/"ground_truth_em.ivecs",heldout_indices,k)
    client=QdrantClient(host="localhost",grpc_port=6334,prefer_grpc=True,timeout=180)
    exists=client.collection_exists(args.collection)
    if exists and args.rebuild:
        client.delete_collection(args.collection)
        exists=False
    build_seconds=0.0
    if not exists:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(size=vectors.shape[1],distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16,ef_construct=100,full_scan_threshold=1_000),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10_000,max_optimization_threads=1),
        )
        start=time.perf_counter()
        upload_dataset(client,args.collection,vectors,labels,args.batch_size)
        client.create_payload_index(collection_name=args.collection,field_name="label",field_schema=models.PayloadSchemaType.INTEGER,wait=True)
        wait_for_index(client,args.collection,3600)
        build_seconds=time.perf_counter()-start
        with open(build_path,"w",encoding="utf-8") as handle:
            json.dump({"build_seconds":build_seconds},handle)
    else:
        wait_for_index(client,args.collection,3600)
        if build_path.is_file():
            with open(build_path,"r",encoding="utf-8") as handle:
                build_seconds=float(json.load(handle)["build_seconds"])
    train_points={}
    if args.exact:
        selected=0
        train_points[selected]=evaluate(client,args.collection,queries,query_labels,train_truth,train_indices,selected,k,1,False,True)
    else:
        ef_values=[16,32,64,128,256,512,1024,2048,4096]
        for ef in ef_values:
            print(f"qdrant training ef={ef}",flush=True)
            train_points[ef]=evaluate(client,args.collection,queries,query_labels,train_truth,train_indices,ef,k,1,args.acorn)
        selected=select_nprobe(train_points,min(1.0,args.target_recall+args.selection_margin),args.safety_factor,ef_values)
        print(f"qdrant selected ef={selected}",flush=True)
    heldout=evaluate(client,args.collection,queries,query_labels,heldout_truth,heldout_indices,selected,k,args.repetitions,args.acorn,args.exact)
    server_version=client.info().version
    row={
        "engine":"qdrant",
        "method":"exact_prefilter" if args.exact else "acorn_filtered_hnsw" if args.acorn else "filtered_hnsw",
        "selected_ef":selected,
        "train_recall":train_points[selected]["recall"],
        **heldout,
        "vector_count":len(vectors),
        "dimension":vectors.shape[1],
        "heldout_queries":len(heldout_indices),
        "target_recall":args.target_recall,
        "build_seconds":build_seconds,
        "storage_megabytes":directory_megabytes(args.storage),
        "server_version":server_version,
        "client_version":"1.18.0",
    }
    result_path=output_path/("qdrant_exact_summary.tsv" if args.exact else "qdrant_acorn_summary.tsv" if args.acorn else "qdrant_summary.tsv")
    with open(result_path,"w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row.keys()),delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row,indent=2))
    print(f"summary={result_path}")


if __name__=="__main__":
    main()
