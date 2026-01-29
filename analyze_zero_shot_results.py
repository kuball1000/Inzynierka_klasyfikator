import json
import csv
import os
from collections import defaultdict
import argparse

def analyze_results(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Archive {input_file} does not exist.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = defaultdict(lambda: {'total': 0, 'correct_ep': 0, 'sum_precision': 0.0})

    for item in data:
        true_ep = item.get('true_ep')
        pred_ep = item.get('pred_ep')
        precision = item.get('precision', 0.0)

        if not true_ep:
            continue

        stats[true_ep]['total'] += 1
        
        if true_ep == pred_ep:
            stats[true_ep]['correct_ep'] += 1
        
        stats[true_ep]['sum_precision'] += precision

    rows = []
    
    header = ["Endpoint", "Total_Count", "Correct_EP_Count", "EP_Accuracy", "Avg_Precision", "Manual_Accuracy_Placeholder"]
    
    for ep in sorted(stats.keys()):
        s = stats[ep]
        total = s['total']
        correct = s['correct_ep']
        ep_acc = correct / total if total > 0 else 0.0
        avg_prec = s['sum_precision'] / total if total > 0 else 0.0
        
        rows.append({
            "Endpoint": ep,
            "Total_Count": total,
            "Correct_EP_Count": correct,
            "EP_Accuracy": f"{ep_acc:.4f}",
            "Avg_Precision": f"{avg_prec:.4f}",
            "Manual_Accuracy_Placeholder": "" 
        })

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    print(f"Analysis complete. Results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze samples_debug_zero_shot.json and export stats to TSV.")
    parser.add_argument("--input", default="samples_debug_zero_shot.json", help="Input JSON file path")
    parser.add_argument("--output", default="analysis_results.tsv", help="Output TSV file path")
    args = parser.parse_args()

    analyze_results(args.input, args.output)
