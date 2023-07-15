from trading_preprocessing import preprocessing

preprocessing('Datasets/SOLUSDT_2022_2023_min.csv', resample='1d',
              profit_ratio=17, horizon=20, plot_prices=True)


