from trading_preprocessing import preprocessing

preprocessing('Datasets/SOLUSDT_2022_2023_min.csv', resample='1D',
              profit_ratio=17, horizon=7, plot_prices=True)