import timeit
import numpy as np
import pandas as pd
import dask.dataframe as dd
from matplotlib import pyplot as plt


def plot(df, figsize=(14, 7), zoom=None, actions=None):
    # Plot the closing prices
    plt.figure(figsize=figsize)

    if zoom:
        # Zoom in between 2900 and 3100
        plt.plot(df.loc[zoom[0]:zoom[1], 'close'], label='Close Price', color='blue')

        if actions:
            # Add points for buying
            buy_points = df[(df['action'] == 'buy') & (df.index >= zoom[0]) & (df.index <= zoom[1])]
            plt.scatter(buy_points.index, buy_points['close'], color='green', label='Buy', marker='^', alpha=1)

            # Add points for selling
            sell_points = df[(df['action'] == 'sell') & (df.index >= zoom[0]) & (df.index <= zoom[1])]
            plt.scatter(sell_points.index, sell_points['close'], color='red', label='Sell', marker='v', alpha=1)

            plt.title(f'Buy and Sell Points (Index {zoom[0]} to {zoom[1]})')

    else:
        plt.plot(df['close'], label='Close Price', color='blue')

        if actions:
            # Add points for buying
            buy_points = df[df['action'] == 'buy']
            plt.scatter(buy_points.index, buy_points['close'], color='green', label='Buy', marker='^', alpha=1)

            # Add points for selling
            sell_points = df[df['action'] == 'sell']
            plt.scatter(sell_points.index, sell_points['close'], color='red', label='Sell', marker='v', alpha=1)

            plt.title('Buy and Sell Points')

    if not actions:
        plt.title('Buy and Sell Points')

    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.legend(loc='best')
    plt.show()


def resample_data(data, freq):
    # Create a copy of the data
    data_copy = data.copy()

    # Sort the data by date in ascending order
    data_copy.sort_values('date', inplace=True)

    # Convert the 'date' column to datetime
    data_copy['date'] = pd.to_datetime(data_copy['date'])

    # Set 'date' as the index
    data_copy.set_index('date', inplace=True)

    # Convert pandas DataFrame to Dask DataFrame
    ddata = dd.from_pandas(data_copy, npartitions=2)

    # Resample to the desired frequency
    open_data = ddata['open'].resample(freq).first().compute()
    high_data = ddata['high'].resample(freq).max().compute()
    low_data = ddata['low'].resample(freq).min().compute()
    close_data = ddata['close'].resample(freq).last().compute()
    volume_data = ddata['volume'].resample(freq).sum().compute()

    # Combine the resampled data
    resampled_data = pd.concat([open_data, high_data, low_data, close_data, volume_data], axis=1)
    resampled_data.columns = ['open', 'high', 'low', 'close', 'volume']

    resampled_data.reset_index(inplace=True)
    resampled_data.dropna(inplace=True)

    return resampled_data


def generate_actions(df, profit_ratio, horizon):
    # Create a new column for actions and initialize it to 'hold'
    df['action'] = 'hold'

    # For each row, check the 'horizon' number of rows in the past
    for i in range(len(df)):
        # Initialize the action as hold
        action = 'hold'
        current_price = df.loc[i, 'close']

        # Check every point within the horizon
        for j in range(1, horizon):

            # Get the price before 'j' number of rows
            past_price = df.loc[i - j, 'close'] if i - j >= 0 else np.NAN

            # Calculate the percentage change from the current price to the past price
            pct_change = (past_price - current_price) / current_price * 100

            future_price = df.loc[i + j, 'close'] if i + j <= len(df) - horizon else np.NAN

            # Calculate the percentage change from the current price to the past price
            fct_change = (future_price - current_price) / future_price * 100

            # If the percentage change is greater than or equal to 'profit_ratio', mark a 'buy' action
            if pct_change >= profit_ratio or fct_change >= profit_ratio:
                action = 'buy'

            # If the percentage change is less than or equal to '-profit_ratio', mark a 'sell' action
            elif pct_change <= -profit_ratio or fct_change <= -profit_ratio:
                action = 'sell'

        # Set the action for the current row
        df.loc[i, 'action'] = action

    return df


def preprocessing(filename, resample=None, profit_ratio=15, horizon=12,
                  plot_prices=False, plot_zoom=None, plot_actions=True):

    start = timeit.default_timer()

    # Load the data
    df = pd.read_csv(filename)

    # Resample the data to different frequencies
    start_time = timeit.default_timer()
    df = resample_data(df, resample)
    end_time = timeit.default_timer()
    print(f"Resampling took {end_time - start_time:.2f} seconds.")

    # Generate actions column (sell, buy, hold)
    start_time = timeit.default_timer()
    df_actions = generate_actions(df, profit_ratio, horizon)
    end_time = timeit.default_timer()
    print(f"adding actions column took {end_time - start_time:.2f} seconds.")

    end = timeit.default_timer()
    print(f"all preprocessing took {end - start:.2f} seconds.")

    # plot
    if plot_prices:
        plot(df_actions, figsize=(14, 7), zoom=plot_zoom, actions=plot_actions)






