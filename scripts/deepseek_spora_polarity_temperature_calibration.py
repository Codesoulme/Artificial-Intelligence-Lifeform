"""Calibrate polarity confidence without changing the classifier weights."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.deepseek_local_baseline import sha
from scripts.deepseek_spora_polarity_gate_probe import OUTPUT as SOURCE

OUTPUT = ROOT / 'runs/deepseek-spora-polarity-temperature-calibration-20261111.json'


def calibrated(rows, temperature):
    result = []
    for row in rows:
        probability = min(max(row['affirmative_probability'], 1e-6), 1 - 1e-6)
        logit = math.log(probability / (1 - probability))
        confidence = 1 / (1 + math.exp(-logit / temperature))
        result.append({**row, 'calibrated_affirmative_probability': confidence,
                       'calibrated_write_confident': row['prediction'] == 0 and confidence >= .9})
    return result


def summary(rows):
    return {'positive_total': sum(row['label'] == 0 for row in rows),
            'positive_write_confident': sum(row['label'] == 0 and row['calibrated_write_confident'] for row in rows),
            'negative_total': sum(row['label'] != 0 for row in rows),
            'negative_write_confident': sum(row['label'] != 0 and row['calibrated_write_confident'] for row in rows)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report_path = SOURCE / 'result.json'
    source_report = json.loads(report_path.read_text(encoding='utf-8'))
    source_sha = sha(report_path)
    calibration = source_report['results']['calibration']['rows']
    labels = torch.tensor([row['label'] == 0 for row in calibration], dtype=torch.float32)
    probabilities = torch.tensor([min(max(row['affirmative_probability'], 1e-6), 1 - 1e-6)
                                  for row in calibration])
    logits = torch.logit(probabilities)
    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.exp())
    calibration_rows = calibrated(calibration, temperature)
    held_rows = calibrated(source_report['results']['held']['rows'], temperature)
    result = {'status': 'development_only_confidence_calibration', 'method': 'temperature_scaling',
              'temperature': temperature, 'write_floor': .9, 'source_report_sha256': source_sha,
              'calibration': summary(calibration_rows), 'held': summary(held_rows),
              'calibration_rows': calibration_rows, 'held_rows': held_rows,
              'limitations': ['Calibration uses the same small synthetic calibration split as the gate development.',
                              'The held split is synthetic and not an independent real dialogue benchmark.',
                              'Temperature scaling changes confidence only; it does not add source, time or authorization semantics.',
                              'This artifact is not connected to persistence.']}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'temperature': temperature, 'calibration': result['calibration'], 'held': result['held']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
