#!/usr/bin/env python3
"""
Standalone training script.
Run: python scripts/train.py [--records 1200]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_full_pipeline
from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Train all Ecomeal AI models")
    parser.add_argument("--records", type=int, default=1200, help="Number of synthetic records")
    parser.add_argument("--no-save", action="store_true", help="Skip saving models to disk")
    args = parser.parse_args()

    logger.info(f"Starting training run: {args.records} records")
    report = run_full_pipeline(
        n_records=args.records,
        save_models=not args.no_save,
        save_data=True,
        generate_report=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Records processed: {report['dataset']['clean_records']}")
    print(f"Wastage model ROC-AUC: {report['wastage_model'].get('roc_auc', 'N/A')}")
    print(f"Wastage model F1: {report['wastage_model'].get('f1', 'N/A')}")
    print(f"Anomalies found: {report['anomaly_detection'].get('n_anomalies_detected', 0)}")
    print(f"Forecast models: {report['forecast_ingredients']}")
    print(f"Critical risk items: {report['portfolio_summary'].get('critical_count', 0)}")
    print(f"High risk items: {report['portfolio_summary'].get('high_risk_count', 0)}")
    print(f"Waste value at risk: ₹{report['portfolio_summary'].get('waste_value_at_risk_inr', 0):,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
