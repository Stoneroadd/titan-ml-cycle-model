"""Fit a tiny standardized linear regression model using only the Python stdlib."""
import argparse
import csv
import json
import math
from pathlib import Path

FEATURES = ["distance_km", "payload_t", "queue_min"]


def read_data(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(FEATURES + ["cycle_time_min"]).issubset(reader.fieldnames):
            raise ValueError("CSV needs distance_km, payload_t, queue_min, cycle_time_min")
        rows = [[float(row[k]) for k in FEATURES] + [float(row["cycle_time_min"])] for row in reader]
    if len(rows) < 6: raise ValueError("At least 6 rows are required")
    if any(not all(math.isfinite(v) for v in row) for row in rows): raise ValueError("Values must be finite")
    return rows


def fit(rows, epochs=2500, rate=0.03):
    split = max(3, int(len(rows) * 0.75))
    train, test = rows[:split], rows[split:]
    means = [sum(r[i] for r in train) / len(train) for i in range(3)]
    scales = [math.sqrt(sum((r[i]-means[i])**2 for r in train)/len(train)) or 1 for i in range(3)]
    weights, bias = [0.0]*3, sum(r[3] for r in train)/len(train)
    for _ in range(epochs):
        grad_w, grad_b = [0.0]*3, 0.0
        for row in train:
            x = [(row[i]-means[i])/scales[i] for i in range(3)]
            error = bias + sum(weights[i]*x[i] for i in range(3)) - row[3]
            grad_b += error
            for i in range(3): grad_w[i] += error*x[i]
        bias -= rate * grad_b / len(train)
        weights = [weights[i] - rate*grad_w[i]/len(train) for i in range(3)]
    predictions = [bias + sum(weights[i]*(r[i]-means[i])/scales[i] for i in range(3)) for r in test]
    actual = [r[3] for r in test]
    mae = sum(abs(a-p) for a,p in zip(actual,predictions))/len(test)
    rmse = math.sqrt(sum((a-p)**2 for a,p in zip(actual,predictions))/len(test))
    mean_y = sum(actual)/len(actual)
    denom = sum((a-mean_y)**2 for a in actual)
    r2 = 1 - sum((a-p)**2 for a,p in zip(actual,predictions))/denom if denom else None
    return {"algorithm":"standardized linear regression / batch gradient descent","synthetic_data":True,
            "train_rows":len(train),"test_rows":len(test),"features":FEATURES,
            "scaler":{"mean":means,"scale":scales},"weights":weights,"bias":bias,
            "metrics":{"mae_min":round(mae,3),"rmse_min":round(rmse,3),"r2":round(r2,3) if r2 is not None else None},
            "test_predictions":[{"actual_min":round(a,2),"predicted_min":round(p,2)} for a,p in zip(actual,predictions)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("input",type=Path);parser.add_argument("--output",type=Path,default=Path("model.json"));args=parser.parse_args()
    try: result=fit(read_data(args.input))
    except (OSError,ValueError) as exc: parser.error(str(exc))
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2))


if __name__=="__main__":main()
