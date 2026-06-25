Experiment notes - raw return baseline

6/25/26 Current setup:
- Target is future return using row shifts, so 10 means 10 trading days.
- Switched default horizon from 5 trading days to 10 trading days.
- Added random_state=42 so XGBoost runs are easier to compare.

6/25/26 What I found:
- The basic classifier is still weak and often predicts mostly Up.
- The more useful signal is in ranked / thresholded predictions, not plain accuracy.
- 10-day predictions looked better than 5-day predictions in the small 10-ticker test.
- Looser regressor settings did not clearly beat the 10-day baseline.
- Best current baseline is the reproducible 10-day version.

6/25/26 Possible next experiments:
- Test on a larger ticker universe.
- Compare predicted returns against SPY-relative returns.
- Evaluate rules that combine classifier probability with regressor ranking.

6/25/26 Combined signal / reporting:
- Added a combined signal report using classifier probability >= 0.60 and top 20% regressor rank.
- Added a readable experiment summary so I do not have to dig through raw training logs every run.
- In the small test, regressor top 20% looked stronger than the combined signal.
- core_25 had some classifier-threshold edge
- core_50 was basically baseline/noise
- broad_200 had a tiny high-confidence classifier pocket but weak regressor edge
- combined signal underperformed
- raw forward return is probably not the final target