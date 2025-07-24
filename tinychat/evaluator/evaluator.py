import json

def elaborate_predictions_and_gt(responses, labels_list):
    """
    Estrae le predizioni e i ground truth da due liste parallele di stringhe JSON.

    Args:
        responses (list of str): lista di risposte JSON del modello
        labels_list (list of str): lista di ground truth JSON

    Returns:
        predictions (list of dict), ground_truth (list of dict)
    """
    predictions = []
    ground_truth = []

    for response, labels in zip(responses, labels_list):
        try:
            parsed_labels = json.loads(labels)
            parsed_response = json.loads(response)
            pred_priority = parsed_response["priority"]  # batch size 1
            pred_availability = parsed_response["availability"]
            gt_priority = parsed_labels["priority"]
            gt_availability = parsed_labels["availability"]

            predictions.append({
                "priority": pred_priority,
                "availability": pred_availability
            })
            ground_truth.append({
                "priority": gt_priority,
                "availability": gt_availability
            })

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue
    #print(f"Extracted {len(predictions)} predictions and {len(ground_truth)} ground truths.")
    return predictions, ground_truth

import re


def extract_json_block(text):
    # Rimuove blocchi markdown tipo ```json ... ```
    text = re.sub(r"```.*?\n", "", text)
    text = re.sub(r"```", "", text)

    brace_count = 0
    start_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_str = text[start_idx:i+1]
                try:
                    # Verifica che sia un JSON valido
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    pass  # Continua a cercare se non è valido

    raise ValueError("No valid JSON object found in text")


import numpy as np
from scipy.stats import kendalltau

class Evaluator:
    def __init__(self, predictions, ground_truth):
        self.predictions = predictions
        self.ground_truth = ground_truth

    def compute_kendall_tau(self):
        kendall_scores = []
        for pred, gt in zip(self.predictions, self.ground_truth):
            # Se pred è una lista (lista di dizionari), prendi il primo elemento
            if isinstance(pred, list):
                pred = pred[0]
            if isinstance(gt, list):
                gt = gt[0]

            pred_priority = pred["priority"]
            #print(f"Debug: {pred_priority}")
            gt_priority = gt["priority"]
            #print(f"Debug: {gt_priority}")

            # Trova intersezione degli ID
            common_ids = list(set(pred_priority) & set(gt_priority))
            #print(f"Debug: Common IDs: {common_ids}")
            # Ordina entrambe le liste mantenendo solo gli elementi comuni
            pred_common = [pid for pid in pred_priority if pid in common_ids]
            gt_common = [pid for pid in gt_priority if pid in common_ids]
            #print(f"Debug: Pred common: {pred_common}")
            #print(f"Debug: GT common: {gt_common}")
            # Controlla se abbiamo almeno due elementi per calcolare kendall tau
            if len(pred_common) < 2:
                tau = 1.0  # non ha senso calcolare tau con <2 elementi
            else:
                tau, _ = kendalltau(gt_common, pred_common)
            kendall_scores.append(tau)
            #print(kendall_scores)
        return np.mean(kendall_scores)


    def compute_availability_accuracy(self):
        all_preds = []
        all_gts = []

        for pred, gt in zip(self.predictions, self.ground_truth):
            if isinstance(pred, list):
                pred = pred[0]
            if isinstance(gt, list):
                gt = gt[0]

            pred_priority = pred["priority"]
            gt_priority = gt["priority"]

            pred_av = pred["availability"]
            gt_av = gt["availability"]

            pred_av_map = {pid: av for pid, av in zip(pred_priority, pred_av)}
            gt_av_map = {pid: av for pid, av in zip(gt_priority, gt_av)}

            ordered_pred_av = [pred_av_map.get(pid, 0) for pid in pred_priority]
            ordered_gt_av = [gt_av_map.get(pid, 0) for pid in pred_priority]

            all_preds.extend(ordered_pred_av)
            all_gts.extend(ordered_gt_av)

        all_preds = np.array(all_preds)
        all_gts = np.array(all_gts)
        accuracy = np.mean(all_preds == all_gts)
        return accuracy

    def compute_confusion_counts(self):
        """
        Conta True Positive, False Positive, False Negative per disponibilità.
        """
        tp = fp = fn = tn = missed_pred = over_pred = 0

        for pred, gt in zip(self.predictions, self.ground_truth):
            if isinstance(pred, list):
                pred = pred[0]
            if isinstance(gt, list):
                gt = gt[0]

            pred_priority = pred["priority"]
            gt_priority = gt["priority"]

            pred_av = pred["availability"]
            gt_av = gt["availability"]

            pred_av_map = {pid: av for pid, av in zip(pred_priority, pred_av)}
            gt_av_map = {pid: av for pid, av in zip(gt_priority, gt_av)}

            all_ids = set(pred_av_map.keys()) | set(gt_av_map.keys())

            for pid in all_ids:
                p = pred_av_map.get(pid, None)
                g = gt_av_map.get(pid, None)
                #print(f"Debug: PID: {pid}, Pred: {p}, GT: {g}")
                if p == 1 and g == 1:
                    tp += 1
                elif p == 1 and g == 0:
                    fp += 1
                elif p == 0 and g == 1:
                    fn += 1
                elif p == 0 and g == 0:
                    tn += 1
                elif p is None:
                    missed_pred += 1
                elif g is None:
                    over_pred += 1

        return tp, fp, fn, tn, missed_pred, over_pred

    def compute_precision_recall_f1(self):
        """
        Calcola precision, recall, F1 basandosi su TP, FP, FN.
        """
        tp, fp, fn, tn, missed_pred, over_pred = self.compute_confusion_counts()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "missed_predictions": missed_pred,
            "over_predictions": over_pred,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    
    

    def evaluate(self):
        return {
            "mean_kendall_tau": self.compute_kendall_tau(),
            "availability_accuracy": self.compute_availability_accuracy(),
            "confusion_counts": self.compute_confusion_counts(),
            "precision_recall_f1": self.compute_precision_recall_f1()
        }
