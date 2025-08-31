import pandas as pd
import os
from typing import Set, Tuple, Any


def safe_split(s: Any) -> Set[str]:
    """Safely split a string by commas, handling nulls and whitespace."""
    if pd.isnull(s) or str(s).strip() == "":
        return set()
    return {x.strip() for x in str(s).split(',') if x.strip()}


# def precision_recall_f1_acc(gt_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float, float, int]:
#     """
#     Compute precision, recall, F1, and set accuracy between two sets.
#     Returns: precision, recall, f1, accuracy, size of gt_set
#     """
#     tp = len(gt_set & pred_set)
#     fp = len(pred_set - gt_set)
#     fn = len(gt_set - pred_set)
#     precision = tp / (tp + fp) if (tp + fp) else 0.0
#     recall = tp / (tp + fn) if (tp + fn) else 0.0
#     f1 = 2 * precision * recall / \
#         (precision + recall) if (precision + recall) else 0.0
#     union = gt_set | pred_set
#     accuracy = len(gt_set & pred_set) / len(union) if union else 1.0
#     return precision, recall, f1, accuracy, len(gt_set)

def precision_recall_f1_acc(
    gt_set: Set[str],
    pred_set: Set[str],
    empty_gt_recall: float = 1.0
) -> Tuple[float, float, float, float, int]:
    """
    Compute precision, recall, F1, and Jaccard Index between two sets.

    Args:
        gt_set: The set of ground truth items.
        pred_set: The set of predicted items.
        empty_gt_recall: The value to return for recall when gt_set is empty.
                         Commonly 1.0 (all zero items were recalled) or 0.0.

    Returns:
        A tuple containing:
        - precision (float)
        - recall (float)
        - f1_score (float)
        - jaccard_index (float)
        - size of gt_set (int)
    """
    if not isinstance(gt_set, set) or not isinstance(pred_set, set):
        raise TypeError("Inputs must be sets.")

    if empty_gt_recall not in [0.0, 1.0]:
        raise ValueError("empty_gt_recall must be 0.0 or 1.0.")

    tp = len(gt_set.intersection(pred_set))

    # Precision = TP / |Predicted Set|
    # If nothing is predicted, precision is 0.
    if not pred_set:
        precision = 0.0
    else:
        precision = tp / len(pred_set)

    # Recall = TP / |Ground Truth Set|
    # If gt_set is empty, the result is ambiguous (0/0).
    # Return the user-specified value.
    if not gt_set:
        recall = empty_gt_recall
    else:
        recall = tp / len(gt_set)

    # F1 Score is the harmonic mean of precision and recall
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    # Jaccard Index = |Intersection| / |Union|
    # If both sets are empty, union is empty. Jaccard is 1.0 (perfect match).
    union_len = len(gt_set.union(pred_set))

    if union_len == 0:
        jaccard_index = 1.0
    else:
        jaccard_index = tp / union_len

    accuracy = jaccard_index

    return precision, recall, f1_score, accuracy, len(gt_set)


# --- File Paths ---
input_file = './A_new/a_new_analysis.xlsx'
output_file = './A_new/a_new_analysis-25-07-16_result_evalBATCH.xlsx'

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

# --- Load Data ---
df = pd.read_excel(input_file)

# --- Ensure Correct Column Names ---
gt_candidates = ['ground_truth_id', 'mitre_attack_id', 'ground_truth']
pred_candidates = ['technique_id_predicted',
                   'technique_id', 'predicted_technique_ids', 'recommendations', 'predicted_techniques']

expected_gt_col = next(
    (col for col in gt_candidates if col in df.columns), None)
expected_pred_col = next(
    (col for col in pred_candidates if col in df.columns), None)

if expected_gt_col is None or expected_pred_col is None:
    print("Available columns:", list(df.columns))
    raise KeyError(
        f"Expected one of {gt_candidates} for ground truth and one of {pred_candidates} for predictions in the input file."
    )

batch_size = 340
num_batches = (len(df) + batch_size - 1) // batch_size

if os.path.exists(output_file):
    print(f"Warning: Output file {output_file} will be overwritten.")

with pd.ExcelWriter(output_file) as writer:
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[start:end].copy()

        # --- Calculate per-sample metrics ---
        per_prec, per_rec, per_f1, per_acc, gt_counts = [], [], [], [], []

        for idx, row in batch_df.iterrows():
            gt = safe_split(row[expected_gt_col])
            pred = safe_split(row[expected_pred_col])
            precision, recall, f1, accuracy, gt_count = precision_recall_f1_acc(
                gt, pred)
            per_prec.append(precision)
            per_rec.append(recall)
            per_f1.append(f1)
            per_acc.append(accuracy)
            gt_counts.append(gt_count)

        batch_df['Precision'] = per_prec
        batch_df['Recall'] = per_rec
        batch_df['F1'] = per_f1
        batch_df['Accuracy'] = per_acc
        batch_df['GT_count'] = gt_counts

        # --- Compute average and weighted metrics ---
        AR = batch_df['Recall'].mean()
        AP = batch_df['Precision'].mean()
        mean_F1 = batch_df['F1'].mean()
        AA = batch_df['Accuracy'].mean()

        total_gt = batch_df['GT_count'].sum() or 1  # avoid division by zero
        WAR = (batch_df['Recall'] * batch_df['GT_count']).sum() / total_gt
        WAP = (batch_df['Precision'] * batch_df['GT_count']).sum() / total_gt
        WAA = (batch_df['Accuracy'] * batch_df['GT_count']).sum() / total_gt

        # --- Print Results for this batch ---
        print(f"\nBatch {batch_idx+1} ({start}:{end}):")
        print(f"Average Recall (AR):               {AR:.4f}")
        print(f"Weighted Average Recall (WAR):     {WAR:.4f}")
        print(f"Average Precision (AP):            {AP:.4f}")
        print(f"Weighted Average Precision (WAP):  {WAP:.4f}")
        print(f"Average F1-Score:                  {mean_F1:.4f}")
        print(f"Average Accuracy (AA):             {AA:.4f}")
        print(f"Weighted Average Accuracy (WAA):   {WAA:.4f}")

        # --- Save metrics to Excel as a separate sheet ---
        metrics_df = pd.DataFrame({
            'Metric': [
                'Average Recall (AR)',
                'Weighted Average Recall (WAR)',
                'Average Precision (AP)',
                'Weighted Average Precision (WAP)',
                'Average F1-Score',
                'Average Accuracy (AA)',
                'Weighted Average Accuracy (WAA)'
            ],
            'Value': [
                AR,
                WAR,
                AP,
                WAP,
                mean_F1,
                AA,
                WAA
            ]
        })

        batch_results_sheet = f'Results_{batch_idx+1}'
        batch_metrics_sheet = f'Metrics_{batch_idx+1}'
        batch_df.to_excel(writer, sheet_name=batch_results_sheet, index=False)
        metrics_df.to_excel(
            writer, sheet_name=batch_metrics_sheet, index=False)
