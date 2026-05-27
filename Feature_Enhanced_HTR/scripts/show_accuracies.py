
import json
from pathlib import Path

def format_percentage(val):
    return f"{val*100:.2f}%"

def show_accuracies():
    results_path = Path("pipeline_results.json")
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run the training pipeline first.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    print("\n" + "="*60)
    print("      HANDWRITTEN TEXT RECOGNITION - MODEL ACCURACIES")
    print("="*60)

    # 1. Training Summary
    train_info = data.get("training", {})
    if train_info:
        print(f"\n[ TRAINING SUMMARY ]")
        print(f"  Total Epochs:  {train_info.get('epochs', 'N/A')}")
        print(f"  Training Samples: {train_info.get('samples', 'N/A')}")
        history = train_info.get("history", {})
        if history.get("loss"):
            print(f"  Final Loss:    {history['loss'][-1]:.4f}")
        if history.get("val_loss"):
            print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")

    # 2. Current Model Performance
    metrics = data.get("metrics", {})
    if metrics:
        print(f"\n[ CURRENT MODEL PERFORMANCE ]")
        print(f"  {'Metric':<20} | {'Before NLP':<12} | {'After NLP':<12}")
        print(f"  {'-'*20}-|-{'-'*12}-|-{'-'*12}")
        
        m_before = metrics.get("before_nlp", {})
        m_after = metrics.get("after_nlp", {})
        
        for key in ["accuracy", "cer", "wer"]:
            name = key.upper() if key != "accuracy" else "Accuracy"
            b_val = m_before.get(key, 0)
            a_val = m_after.get(key, 0)
            
            # Format as percentage for accuracy, decimal for others
            if key == "accuracy":
                b_str = format_percentage(b_val)
                a_str = format_percentage(a_val)
            else:
                b_str = f"{b_val:.4f}"
                a_str = f"{a_val:.4f}"
            
            print(f"  {name:<20} | {b_str:<12} | {a_str:<12}")

    # 3. Baseline Comparison
    comparison = data.get("comparison", {}).get("baseline_models", {})
    if comparison:
        print(f"\n[ BASELINE COMPARISON (Historical) ]")
        print(f"  {'Model Architecture':<30} | {'Accuracy':<10} | {'CER':<10}")
        print(f"  {'-'*30}-|-{'-'*10}-|-{'-'*10}")
        
        for model_name, stats in comparison.items():
            acc = format_percentage(stats.get("accuracy", 0))
            cer = f"{stats.get('cer', 0):.2f}"
            print(f"  {model_name:<30} | {acc:<10} | {cer:<10}")

    # 4. Custom Evaluation Results (if any)
    custom_eval = data.get("custom_evaluation", [])
    if custom_eval:
        print(f"\n[ CUSTOM IMAGE EVALUATION ]")
        print(f"  Total Custom Images: {len(custom_eval)}")
        correct = sum(1 for item in custom_eval if item['ground_truth'].lower() == item['final'].lower())
        custom_acc = correct / len(custom_eval)
        print(f"  Custom Accuracy:     {format_percentage(custom_acc)}")

    print("\n" + "="*60)

if __name__ == "__main__":
    show_accuracies()
