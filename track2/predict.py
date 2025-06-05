import argparse
import numpy as np
import pandas as pd
import scipy.stats
from audiobox_aesthetics.infer import initialize_predictor

def evaluate_predictions(
    csv_path: str,
    ckpt: str = None,
    batch_size: int = 1,
    output_csv_path: str = "predictions_with_scores.csv"
) -> dict:
    """
    讀取 csv，對所有 data_path 做推論，並計算 MSE/LCC/SRCC/KTAU。

    Args:
        csv_path (str): 輸入 CSV 路徑，需包含 sample_id, data_path, 四個評分欄位。
        ckpt (str, optional): 模型 checkpoint 路徑，若提供則載入該 ckpt。
        batch_size (int): 推論批次大小。
        output_csv_path (str): 預測結果輸出 CSV 路徑。

    Returns:
        dict: 每個面向的四個指標。
    """
    # 1) 讀取 CSV
    df = pd.read_csv(csv_path)

    # 2) 初始化預測器
    predictor = initialize_predictor(ckpt=ckpt) if ckpt else initialize_predictor()

    # 3) 準備並執行批次推論
    inputs = [{"path": path} for path in df['data_path']]
    predictions = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        results = predictor.forward(batch_inputs)
        predictions.extend(results)

    # 4) 收集預測結果
    pred_CE = np.array([pred['CE'] for pred in predictions])
    pred_CU = np.array([pred['CU'] for pred in predictions])
    pred_PC = np.array([pred['PC'] for pred in predictions])
    pred_PQ = np.array([pred['PQ'] for pred in predictions])

    # 5) 收集真實分數
    true_CE = df['Content_Enjoyment'].to_numpy()
    true_CU = df['Content_Usefulness'].to_numpy()
    true_PC = df['Production_Complexity'].to_numpy()
    true_PQ = df['Production_Quality'].to_numpy()

    # 6) 儲存含預測分數的 CSV
    df['pred_CE'] = pred_CE
    df['pred_CU'] = pred_CU
    df['pred_PC'] = pred_PC
    df['pred_PQ'] = pred_PQ
    df.to_csv(output_csv_path, index=False)

    # 7) 指標計算
    def _eval(true, pred):
        mse = np.mean((true - pred) ** 2)
        lcc = np.corrcoef(true, pred)[0, 1]
        srcc = scipy.stats.spearmanr(true, pred)[0]
        ktau = scipy.stats.kendalltau(true, pred)[0]
        return {"MSE": mse, "LCC": lcc, "SRCC": srcc, "KTAU": ktau}

    metrics = {
        "CE": _eval(true_CE, pred_CE),
        "CU": _eval(true_CU, pred_CU),
        "PC": _eval(true_PC, pred_PC),
        "PQ": _eval(true_PQ, pred_PQ),
    }
    return metrics


def save_metrics_to_csv(metrics: dict, save_path: str):
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.to_csv(save_path, index_label='Metric')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate AudioMOS model predictions and compute metrics'
    )
    parser.add_argument('input_csv', help='輸入 CSV 路徑')
    parser.add_argument('output_csv', help='含預測分數的輸出 CSV 路徑')
    parser.add_argument('metrics_csv', help='指標結果的輸出 CSV 路徑')
    parser.add_argument('--ckpt', default=None, help='模型 checkpoint 路徑')
    parser.add_argument('--batch_size', type=int, default=1, help='推論批次大小')
    args = parser.parse_args()

    metrics = evaluate_predictions(
        csv_path=args.input_csv,
        ckpt=args.ckpt,
        batch_size=args.batch_size,
        output_csv_path=args.output_csv
    )
    save_metrics_to_csv(metrics, args.metrics_csv)
    print(f"✅ 完成推論，預測結果: {args.output_csv}, 指標結果: {args.metrics_csv}")
