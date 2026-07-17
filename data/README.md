# Training data

```
data/
  carrier-sample.mp4       # optional cover for vsteg synthetic samples
  payload-sample.txt       # optional secret for vsteg synthetic samples
  tools/
    <tool-name>/           # e.g. openpuff/
      original-1.mp4       # clean / cover (also accepts original1.mp4)
      stego-1.mp4          # stego twin
      target-1.txt         # optional payload (not required to train)
      original-2.mp4
      stego-2.mp4
      target-2.txt
```

Train:

```bash
pip install -e ".[ml]"
python scripts/train_ml_detector.py
```

Writes `models/runs/<timestamp>.joblib` and replaces `models/latest-model`.
