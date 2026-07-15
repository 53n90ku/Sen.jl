import argparse
import csv
import hashlib
import json
import resource
import time
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path

import faiss
import numpy as np


def load_fvecs(path):
    raw=np.memmap(path,dtype=np.int32,mode="r")
    dimension=int(raw[0])
    records=raw.reshape(-1,dimension+1)
    if not np.all(records[:,0]==dimension):
        raise ValueError("fvecs dimensions do not match")
    return records[:,1:].view(np.float32)


def load_database_vectors(path):
    f32_path=path/"database_vectors.f32"
    if f32_path.is_file():
        with open(path/"sen_dataset.toml","rb") as handle:
            manifest=tomllib.load(handle)
        count=int(manifest["vector_count"])
        dimension=int(manifest["dimension"])
        return np.memmap(f32_path,dtype=np.float32,mode="r",shape=(count,dimension))
    return load_fvecs(path/"database_vectors.fvecs")


def load_labels(path):
    labels=[]
    with open(path,"r",encoding="utf-8") as handle:
        for line in handle:
            labels.append(int(json.loads(line)["number_of_sub_categories"]))
    return np.asarray(labels,dtype=np.int64)


def load_truth(path,selected,k):
    wanted={int(index):position for position,index in enumerate(selected)}
    truth=[None]*len(selected)
    with open(path,"rb") as handle:
        source_index=0
        while wanted:
            dimension_value=np.fromfile(handle,dtype=np.int32,count=1)
            if len(dimension_value)==0:
                break
            dimension=int(dimension_value[0])
            values=np.fromfile(handle,dtype=np.int32,count=dimension)
            if len(values)!=dimension:
                raise ValueError("ivecs record is truncated")
            if source_index in wanted:
                truth[wanted.pop(source_index)]=values[:k].astype(np.int64)
            source_index+=1
    if wanted:
        raise ValueError("groundtruth indices are missing")
    return truth


def load_cached_truth(prefix,selected,k):
    with open(f"{prefix}.toml","rb") as handle:
        manifest=tomllib.load(handle)
    source_indices=[int(value)-1 for value in manifest["query_indices"]]
    positions={value:index for index,value in enumerate(source_indices)}
    missing=[int(value) for value in selected if int(value) not in positions]
    if missing:
        raise ValueError("groundtruth cache does not contain selected queries")
    raw=np.fromfile(f"{prefix}.ivecs",dtype=np.int32)
    width=int(manifest["k"])+1
    records=raw.reshape(-1,width)
    if not np.all(records[:,0]==int(manifest["k"])):
        raise ValueError("groundtruth cache dimensions do not match")
    values=records[:,1:]
    return [values[positions[int(index)],:k].astype(np.int64) for index in selected]


def load_query_indices(sen_environment=None,query_partitions=None,k=10):
    if sen_environment:
        with open(sen_environment,"rb") as handle:
            environment=tomllib.load(handle)
        return(
            np.asarray(environment["train_query_indices"],dtype=np.int64)-1,
            np.asarray(environment["heldout_query_indices"],dtype=np.int64)-1,
            int(environment["k"]),
        )
    public_path=Path(f"{query_partitions}.toml")
    sealed_path=Path(f"{query_partitions}.confirmation.toml")
    with open(public_path,"rb") as handle:
        public=tomllib.load(handle)
    with open(sealed_path,"rb") as handle:
        sealed=tomllib.load(handle)
    confirmation=[int(value) for value in sealed["confirmation_indices"]]
    digest=hashlib.sha256(",".join(str(value) for value in confirmation).encode()).hexdigest()
    if digest!=public["confirmation_hash"] or digest!=sealed["confirmation_hash"]:
        raise ValueError("confirmation partition hash changed")
    return(
        np.asarray(public["development_indices"],dtype=np.int64)-1,
        np.asarray(confirmation,dtype=np.int64)-1,
        int(k),
    )


def recall_at_k(predicted,truth,k):
    predicted={int(value) for value in predicted[:k] if value>=0}
    expected=[int(value) for value in truth[:k] if value>=0]
    if not expected:
        return 1.0 if not predicted else 0.0
    return len(predicted.intersection(expected))/len(expected)


def percentile(values,percentile_value):
    return float(np.percentile(np.asarray(values,dtype=np.float64),percentile_value,method="higher"))


def evaluate_ivf(index,vectors,queries,query_labels,labels,truth,indices,nprobe,k,repetitions):
    recalls=[]
    times=[]
    for query_position,source_index in enumerate(indices):
        query=np.ascontiguousarray(queries[source_index:source_index+1])
        matching=np.flatnonzero(labels==query_labels[source_index]).astype(np.int64)
        selector=faiss.IDSelectorBatch(matching)
        parameters=faiss.SearchParametersIVF(nprobe=nprobe,sel=selector)
        index.search(query,k,params=parameters)
        latest=None
        for _ in range(repetitions):
            start=time.perf_counter_ns()
            _,latest=index.search(query,k,params=parameters)
            times.append((time.perf_counter_ns()-start)/1_000_000)
        recalls.append(recall_at_k(latest[0],truth[query_position],k))
    return {
        "recall":float(np.mean(recalls)),
        "p50_ms":percentile(times,50),
        "p95_ms":percentile(times,95),
        "qps":1000/float(np.mean(times)),
    }


def evaluate_exact(vectors,queries,query_labels,labels,truth,indices,k,repetitions):
    recalls=[]
    times=[]
    for query_position,source_index in enumerate(indices):
        query=np.asarray(queries[source_index])
        matching=np.flatnonzero(labels==query_labels[source_index])
        matching_vectors=np.ascontiguousarray(vectors[matching])
        latest=None
        for repetition in range(repetitions+1):
            start=time.perf_counter_ns()
            scores=matching_vectors@query
            count=min(k,len(scores))
            if count==0:
                latest=np.empty(0,dtype=np.int64)
            else:
                positions=np.argpartition(scores,-count)[-count:]
                positions=positions[np.argsort(scores[positions])[::-1]]
                latest=matching[positions]
            elapsed=(time.perf_counter_ns()-start)/1_000_000
            if repetition>0:
                times.append(elapsed)
        recalls.append(recall_at_k(latest,truth[query_position],k))
    return {
        "recall":float(np.mean(recalls)),
        "p50_ms":percentile(times,50),
        "p95_ms":percentile(times,95),
        "qps":1000/float(np.mean(times)),
    }


def select_nprobe(points,target,safety_factor,nprobes):
    eligible=[nprobe for nprobe in nprobes if points[nprobe]["recall"]>=target]
    if eligible:
        selected=min(eligible)
    else:
        selected=max(nprobes,key=lambda value:(points[value]["recall"],-value))
    target_probe=min(max(nprobes),int(np.ceil(selected*safety_factor)))
    return min(value for value in nprobes if value>=target_probe)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True)
    split_group=parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--sen-environment")
    split_group.add_argument("--query-partitions")
    parser.add_argument("--output",required=True)
    parser.add_argument("--truth-cache")
    parser.add_argument("--nlists",type=int,default=64)
    parser.add_argument("--training-count",type=int,default=20_000)
    parser.add_argument("--target-recall",type=float,default=0.95)
    parser.add_argument("--selection-margin",type=float,default=0.02)
    parser.add_argument("--safety-factor",type=float,default=2.0)
    parser.add_argument("--repetitions",type=int,default=2)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--k",type=int,default=10)
    args=parser.parse_args()
    faiss.omp_set_num_threads(1)
    data_path=Path(args.data)
    output_path=Path(args.output)
    output_path.mkdir(parents=True,exist_ok=True)
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
    quantizer=faiss.IndexFlatIP(vectors.shape[1])
    index=faiss.IndexIVFFlat(quantizer,vectors.shape[1],args.nlists,faiss.METRIC_INNER_PRODUCT)
    rng=np.random.default_rng(args.seed)
    training_indices=rng.choice(len(vectors),size=min(args.training_count,len(vectors)),replace=False)
    start=time.perf_counter()
    index.train(np.ascontiguousarray(vectors[training_indices]))
    index.add_with_ids(np.ascontiguousarray(vectors),np.arange(len(vectors),dtype=np.int64))
    build_seconds=time.perf_counter()-start
    nprobes=[]
    value=1
    while value<args.nlists:
        nprobes.append(value)
        value*=2
    nprobes.extend(int(np.ceil(args.nlists*fraction)) for fraction in [0.125,0.25,0.375,0.5,0.75])
    nprobes.append(args.nlists)
    nprobes=sorted(set(nprobes))
    train_points={}
    for nprobe in nprobes:
        print(f"faiss training nprobe={nprobe}",flush=True)
        train_points[nprobe]=evaluate_ivf(index,vectors,queries,query_labels,labels,train_truth,train_indices,nprobe,k,1)
    selected=select_nprobe(train_points,min(1.0,args.target_recall+args.selection_margin),args.safety_factor,nprobes)
    print(f"faiss selected nprobe={selected}",flush=True)
    exact=evaluate_exact(vectors,queries,query_labels,labels,heldout_truth,heldout_indices,k,args.repetitions)
    ivf=evaluate_ivf(index,vectors,queries,query_labels,labels,heldout_truth,heldout_indices,selected,k,args.repetitions)
    rows=[
        {"engine":"faiss","method":"exact_prefilter","selected_nprobe":0,"train_recall":1.0,**exact},
        {"engine":"faiss","method":"ivf_prefilter","selected_nprobe":selected,"train_recall":train_points[selected]["recall"],**ivf},
    ]
    for row in rows:
        row.update({
            "vector_count":len(vectors),
            "dimension":vectors.shape[1],
            "heldout_queries":len(heldout_indices),
            "target_recall":args.target_recall,
            "build_seconds":build_seconds,
            "peak_rss_megabytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024,
            "threads":1,
            "faiss_version":faiss.__version__,
        })
    result_path=output_path/"faiss_summary.tsv"
    with open(result_path,"w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0].keys()),delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows,indent=2))
    print(f"summary={result_path}")


if __name__=="__main__":
    main()
