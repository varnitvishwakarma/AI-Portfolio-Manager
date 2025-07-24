# AI Portfolio Project

This project demonstrates **Artificial Intelligence in Quantitative Finance** to build smarter portfolio management and asset allocation strategies.

We train two deep learning models on historical financial data:

- ✅ **LSTM** (Long Short-Term Memory)
- ✅ **Transformer**

We then use their predicted returns to dynamically allocate portfolio weights and evaluate performance.



## 📂 Project Structure

AI_Portfolio_Project/
│
├── data/
│ ├── raw/ # Raw CSVs
│ └── processed/ # Processed CSVs with features
│
├── notebooks/ # Jupyter notebooks for experiments
│ ├── 01_data_collection.ipynb
│ ├── 02_lstm_model.ipynb
│ ├── 03_transformer_model.ipynb
│ └── 04_portfolio_optimization.ipynb
│
├── results/ # Saved models & outputs
│ ├── lstm_model.h5 / lstm_model.keras
│ ├── transformer_model.h5 / transformer_model.keras
│ ├── X_test.npy / y_test.npy
│ ├── daily_ret_lstm.npy / daily_ret_trans.npy
│
├── scripts/ # Main Python scripts
│ ├── data_collection.py # Collect raw data
│ ├── preprocessing.py # Preprocess and feature engineering
│ ├── models_lstm.py # Train LSTM model
│ ├── models_transformer.py # Train Transformer model
│ ├── portfolio_strategy.py # Generate portfolio weights and returns
│ ├── run_project.py # Run models and print metrics
│ └── metrics.py # Calculate performance metrics
│
├── requirements.txt
├── .gitignore
└── README.md





## ⚙️ Requirements

Create a virtual environment and install dependencies:


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
🚀 How to Run
📌 1. Collect Data

python scripts/data_collection.py
📌 2. Preprocess Data

python scripts/preprocessing.py
📌 3. Train LSTM Model

python scripts/models_lstm.py
📌 4. Train Transformer Model

python scripts/models_transformer.py
📌 5. Run Portfolio Strategy & Print Metrics

python scripts/run_project.py
Metrics such as Annualized Return, Volatility, Sharpe Ratio, and Max Drawdown will be printed in the console.

📊 Results
Outputs are saved in the results/ folder:

lstm_model.keras and transformer_model.keras

X_test.npy / y_test.npy

daily_ret_lstm.npy / daily_ret_trans.npy

📈 (Optional) Plot Results
You can create a script named plot_results.py (or add to your notebooks) to visualize:

Equity Curves (Cumulative Returns)

Drawdowns

Example plotting code is available in the discussion above.

✨ Notes
Both models use lookback = 60.

All data is aligned on common dates across assets.

You can tweak hyperparameters inside the training scripts.

📧 Contact
For questions or suggestions:

Author: Varnit Vishwakarma

Enjoy exploring smarter portfolio management! 🚀📈🔥



✅ **Let me know if you want me to also generate `plot_results.py` in this repo (or integrate plotting into a notebook)!**  
🔥 You’re very close to final submission—just ask if you need anything else like the PDF report template or additional polish.







