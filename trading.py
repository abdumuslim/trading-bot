from trading_preprocessing import preprocessing
from trading_model import trading_model

time_steps = 128
models_list = ["Dense", "ensemble"]

df = preprocessing('Datasets/SOLUSDT_2022_2023_min2.csv', resample='1h',
                   profit_ratio=10, horizon=128, indicators=['all'], plot_prices=False, plot_zoom=None,
                   chart_type='candle', chart_patterns=False)


# df.to_csv('Datasets/processed_data1.csv', index=False)

trading_model(df, use_model=models_list[0], batch_size=time_steps, epochs=600, test_size=0.2, time_steps=time_steps)


