# Trading Bot

This repository contains the code for a simple trading bot. The bot is designed to automate the process of buying and selling stocks based on predefined rules. The project is structured into three main Python scripts: `trading.py`, `trading_preprocessing.py`, and `trading_model.py`.

## Files Description

### 1. trading.py

This script is the main entry point of the trading bot. It is responsible for executing the trading operations based on the predictions made by the model. It interacts with the stock market, places orders, and manages the portfolio.

### 2. trading_preprocessing.py

This script is responsible for preprocessing the trading data. The preprocessing steps include:

- Dropping any rows with missing values.
- Creating a new column 'HL_PCT' which represents the percentage change between the High and Low price of the stock for a day.
- Creating another new column 'PCT_change' which represents the percentage change in the Close price from the Open price of the stock for a day.
- Dropping 'Open', 'High', 'Low', 'Volume', 'Adj Close' columns from the DataFrame.
- Scaling the 'Close', 'HL_PCT', and 'PCT_change' columns using MinMaxScaler.

The script also includes functions to create sequences from the preprocessed data and to split the data into training and test sets.

### 3. trading_model.py

This script is responsible for defining and training the model. The model is trained on the preprocessed data and then used to make predictions on the test data. The predictions are then used by the `trading.py` script to make trading decisions.

## Getting Started

To get started with this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/abdumuslim/trading-bot.git
cd trading-bot
pip install -r requirements.txt
```

After installing the dependencies, you can run the `trading.py` script to start the trading bot.

```bash
python trading.py
```

Please note that you might need to provide your own API keys for the stock market data, depending on the data source used in the scripts.

## Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
