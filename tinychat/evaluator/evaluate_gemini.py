import os
from tinychat.evaluator.evaluator import Evaluator, elaborate_predictions_and_gt
from tinychat.evaluator.evaluator import extract_json_block
import json
import argparse

def main():
    # Load environment variables
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument("--prediction-gemini-label-folder", type=str, default=["/home/workspace/social_interaction_dataset/social_interaction_dataset_test_gemini/labels/", "/home/workspace/social_interaction_dataset_1/social_interaction_dataset_1_test_gemini/labels/"])
    parser.add_argument("--gt-label-folder", type=str, default=["/home/workspace/social_interaction_dataset/social_interaction_dataset_test/labels/", "/home/workspace/social_interaction_dataset_1/social_interaction_dataset_1_test/labels/"])
    args = parser.parse_args()
    all_predictions = []
    all_ground_truth = []
    for folder in args.prediction_gemini_label_folder:
        for root, _, pred_label_file in os.walk(folder):
            for pred_label in pred_label_file:
                if pred_label.endswith(".txt"):
                    pred_label_full_path = os.path.join(root, pred_label)
                    if "social_interaction_dataset_test_gemini" in pred_label_full_path:
                        label_full_path = pred_label_full_path.replace("social_interaction_dataset_test_gemini", "social_interaction_dataset_test")
                    elif "social_interaction_dataset_1_test_gemini" in pred_label_full_path:
                        label_full_path = pred_label_full_path.replace("social_interaction_dataset_1_test_gemini", "social_interaction_dataset_1_test")
                    #print(f"Processing {pred_label_full_path}")
                    with open(pred_label_full_path, "r") as f:
                        content = f.read()
                    try:
                        json_str_response = extract_json_block(content)
                    except Exception as e:
                        print(f"Error decoding JSON from {pred_label_full_path}: {e}")
                        continue
                    with open(label_full_path, "r") as f:
                        content = f.read()
                    try:
                        json_str_label = extract_json_block(content)
                    except Exception as e:
                        print(f"Error decoding JSON from {label_full_path}: {e}")
                        continue
                    predictions, ground_truth = elaborate_predictions_and_gt([json_str_response], [json_str_label])
                    #print(f"Predictions: {predictions}")
                    #print(f"Ground Truth: {ground_truth}")
                    all_predictions.append(predictions)
                    all_ground_truth.append(ground_truth)
    evaluator = Evaluator(all_predictions, all_ground_truth)
    results = evaluator.evaluate()
    print(f"Evaluation results: {results}")
    
if __name__ == "__main__":
    main()