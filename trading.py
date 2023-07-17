from trading_preprocessing import preprocessing

preprocessing('Datasets/SOLUSDT_2022_2023_min.csv', resample='1h',
              profit_ratio=5, horizon=24, indicators=['all'], plot_prices=True, plot_zoom=[5200, 5300],
              chart_type='candle', chart_patterns=False)


