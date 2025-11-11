# AI CUP 2025 玉山人工智慧公開挑戰賽－AI偵探出任務，精準揪出警示帳戶！
## 使用方式
1. **安裝套件**
   - 請先安裝 Python 3.8或以上，及下列套件：
     - pandas==2.0.0
     - scikit-learn==1.3.2
     - numpy==1.26.4
     - catboost==1.2.8
     - xgboost==3.1.1
2. **準備資料**
   - 請將以下三個CSV檔案放在 `dir_path` 指定的資料夾：
     - acct_transaction.csv
     - acct_alert.csv
     - acct_predict.csv
   - 若資料路徑不同，請修改程式中的 `dir_path` 變數。
3. **執行程式**
   ```powershell
   python TransactionAlertClassifier.py --data-dir "T-Brain Competition Preliminary Data V3/初賽資料" --output result.csv --probs-output probs.csv --obs-day 105 --future-window 16 --neg-multiplier 10 --folds 5 --target-positive-rate -1
   ```
   - 預測結果將會儲存於 result.csv。
