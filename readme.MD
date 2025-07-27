# AI Portfolio Project

This project demonstrates **Artificial Intelligence in Quantitative Finance** to build smarter portfolio management and asset allocation strategies.

We train two deep learning models on historical financial data:

- ✅ **LSTM** (Long Short-Term Memory)
- ✅ **Transformer**

We then use their predicted returns to dynamically allocate portfolio weights and evaluate performance.


## ⚙️ Requirements

#### Create a virtual environment and install dependencies:

pip install -r requirements.txt

requirements.txt

nginx
Copy
Edit
pandas
numpy
tensorflow
scikit-learn
matplotlib


### 🚀 How to Run
#### 📌 1. Collect Data
python scripts/data_collection.py


#### 📌 2. Preprocess Data
python scripts/preprocessing.py

#### 📌 3. Train LSTM Model
python scripts/models_lstm.py

#### 📌 4. Train Transformer Model
python scripts/models_transformer.py

#### 📌 5. Run Portfolio Strategy & Print Metrics
python scripts/run_evaulate.py


Metrics such as Annualized Return, Volatility, Sharpe Ratio, and Max Drawdown will be printed in the console.

## 📊 Results
Outputs are saved in the results/ folder:

lstm_model.keras and transformer_model.keras

X_test.npy / y_test.npy

daily_ret_lstm.npy / daily_ret_trans.npy

## 📈 Plot Results
You can create a script named plot_results.py (or add to your notebooks) to visualize:

Equity Curves (Cumulative Returns)

Drawdowns

Example plotting code is available in the discussion above.

📧 Contact
For questions or suggestions:

Author: Varnit Vishwakarma
email: varnitusar25@gmail.com

Enjoy exploring smarter portfolio management! 🚀📈🔥



✅ **Let me know if you want me to also generate `plot_results.py` in this repo (or integrate plotting into a notebook)!**  
🔥 You’re very close to final submission—just ask if you need anything else like the PDF report template or additional polish.







