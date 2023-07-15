from trading_preprocessing import preprocessing

preprocessing('Datasets/SOLUSDT_2022_2023_min.csv', resample='1h',
              profit_ratio=17, horizon=20, indicators=None, plot_prices=True, plot_zoom=[2000, 2600])


