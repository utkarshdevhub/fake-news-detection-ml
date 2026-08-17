# Dataset Instructions

Use the instructor-approved labeled fake/real news benchmark.

Expected local files:
- `data/Fake.csv`
- `data/True.csv`

The CSV files are excluded from Git using `.gitignore`. This keeps the
repository lightweight and avoids redistributing dataset contents without
checking the dataset's terms.

After adding the files locally, run:

```bash
python train.py
```
