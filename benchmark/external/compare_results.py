import argparse
import csv
import json
from pathlib import Path


def read_rows(path):
    with open(path,"r",encoding="utf-8") as handle:
        return list(csv.DictReader(handle,delimiter="\t"))


def normalized_row(engine,row,target):
    recall=float(row.get("heldout_recall",row.get("recall")))
    return {
        "engine":engine,
        "method":row["method"],
        "recall_at_10":recall,
        "p50_ms":float(row["p50_ms"]),
        "p95_ms":float(row["p95_ms"]),
        "qps":float(row["qps"]) if row.get("qps") else "",
        "comparable_recall":recall>=target,
        "build_seconds":float(row.get("build_seconds",0)),
        "peak_rss_megabytes":row.get("peak_rss_megabytes",row.get("peak_rss_megabytes","")),
    }


def sen_qps_by_method(path):
    rows=read_rows(path)
    groups={"hybrid":rows}
    for bucket in ["rare","medium","broad"]:
        groups[f"hybrid_{bucket}"]=[row for row in rows if row.get("bucket")==bucket]
    qps={}
    for method,values in groups.items():
        if values:
            mean_latency=sum(float(row["latency_ms"]) for row in values)/len(values)
            qps[method]=1000/mean_latency
    return qps


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sen",required=True)
    parser.add_argument("--external",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--target-recall",type=float,default=0.95)
    parser.add_argument("--sen-build-seconds",type=float,default=0.0)
    parser.add_argument("--sen-raw")
    args=parser.parse_args()
    external=Path(args.external)
    with open(external/"qdrant_build.json","r",encoding="utf-8") as handle:
        qdrant_build_seconds=float(json.load(handle)["build_seconds"])
    rows=[]
    sen_qps=sen_qps_by_method(args.sen_raw) if args.sen_raw else {}
    for row in read_rows(args.sen):
        normalized=normalized_row("sen",row,args.target_recall)
        normalized["build_seconds"]=args.sen_build_seconds
        normalized["qps"]=sen_qps.get(row["method"],normalized["qps"])
        rows.append(normalized)
    for filename,engine in [
        ("faiss_summary.tsv","faiss"),
        ("qdrant_summary.tsv","qdrant"),
        ("qdrant_acorn_summary.tsv","qdrant"),
        ("qdrant_exact_summary.tsv","qdrant"),
    ]:
        for row in read_rows(external/filename):
            normalized=normalized_row(engine,row,args.target_recall)
            if engine=="qdrant":
                normalized["build_seconds"]=qdrant_build_seconds
            rows.append(normalized)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with open(output,"w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0].keys()),delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("engine\tmethod\trecall\tp50_ms\tp95_ms\tcomparable")
    for row in rows:
        print(f"{row['engine']}\t{row['method']}\t{row['recall_at_10']:.4f}\t{row['p50_ms']:.4f}\t{row['p95_ms']:.4f}\t{row['comparable_recall']}")
    print(f"comparison={output}")


if __name__=="__main__":
    main()
