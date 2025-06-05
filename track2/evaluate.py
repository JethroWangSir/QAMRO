import argparse
import numpy as np
import pandas as pd
from audiobox_aesthetics.infer import initialize_predictor

def evaluate_predictions(
    eval_list_path: str,
    wav_dir: str,
    ckpt: str = None,
    batch_size: int = 1,
    output_csv_path: str = "answer.txt"
) -> None:
    """
    Perform inference using AudioMOS model on evaluation audio list.

    Args:
        eval_list_path (str): Path to eval_list.txt, each line is a sample_id.
        wav_dir (str): Directory containing .wav files.
        ckpt (str, optional): Checkpoint to load.
        batch_size (int): Inference batch size.
        output_csv_path (str): Path to save predictions (answer.txt).
    """
    # 1. Load eval list
    with open(eval_list_path, 'r') as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    # 2. Construct full paths
    data_paths = [f"{wav_dir}/{sample_id}.wav" for sample_id in sample_ids]
    df = pd.DataFrame({'sample_id': sample_ids, 'data_path': data_paths})

    # 3. Initialize predictor
    predictor = initialize_predictor(ckpt=ckpt) if ckpt else initialize_predictor()

    # 4. Run inference
    inputs = [{"path": path} for path in df['data_path']]
    predictions = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        results = predictor.forward(batch_inputs)
        predictions.extend(results)

    # 5. Save results to CSV
    df_preds = pd.DataFrame(predictions)
    df_preds.insert(0, 'sample_id', sample_ids)  # Insert sample_id as first column

    # Reorder columns
    new_order = ["PQ", "PC", "CE", "CU"]
    existing_columns = [col for col in new_order if col in df_preds.columns]
    other_columns = [col for col in df_preds.columns if col not in existing_columns and col != 'sample_id']
    df_preds = df_preds[['sample_id'] + existing_columns + other_columns]

    df_preds.to_csv(output_csv_path, index=False)

    print(f"✅ Inference completed. Predictions saved to: {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AudioMOS evaluation phase inference')
    parser.add_argument('--eval_list', required=True, help='Path to eval_list.txt')
    parser.add_argument('--wav_dir', required=True, help='Directory with .wav files')
    parser.add_argument('--output_csv', required=True, help='Output file (e.g., answer.txt)')
    parser.add_argument('--ckpt', default=None, help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=1, help='Inference batch size')
    args = parser.parse_args()

    evaluate_predictions(
        eval_list_path=args.eval_list,
        wav_dir=args.wav_dir,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        output_csv_path=args.output_csv
    )