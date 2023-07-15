import timeit
import numpy as np
import pandas as pd
import dask.dataframe as dd
from matplotlib import pyplot as plt, gridspec


def plot(df, figsize=(14, 7), zoom=None, actions=None, indicators=None):
    """
    Plots the closing prices and optionally highlights the points of action (buy or sell).
    Can also plot the specified indicators.

    Args:
    df (DataFrame): Data to plot.
    figsize (tuple): Figure size.
    zoom (list): Range of indices to zoom in on (ex: [200, 300]).
    actions (bool): Whether to plot the points of action.
    indicators (list): List of indicators to plot from ['ATR', 'MA_20', 'MA_100', 'MA_200', 'RSI'].
    """

    # List of all possible indicators
    all_indicators = ['ATR', 'MA_20', 'MA_100', 'MA_200', 'RSI']

    # If no specific indicators are provided, show all
    if indicators is None:
        indicators = all_indicators

    # Separate MA indicators to plot on the main chart
    ma_indicators = [ind for ind in indicators if "MA" in ind]
    other_indicators = [ind for ind in indicators if ind not in ma_indicators]

    # Create a gridspec with the specified heights
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 2, 1, 1])

    fig = plt.figure(figsize=figsize)

    # Create the subplots
    ax_atr = plt.subplot(gs[0])
    ax_price = plt.subplot(gs[1], sharex=ax_atr)
    ax_rsi = plt.subplot(gs[2], sharex=ax_atr)

    # Hide x labels and tick labels for all but bottom plot
    plt.setp(ax_atr.get_xticklabels(), visible=False)
    plt.setp(ax_price.get_xticklabels(), visible=False)

    # Plot close prices and MA indicators
    if zoom:
        ax_price.plot(df.loc[zoom[0]:zoom[1], 'close'], label='Close Price', color='blue')
        for ma in ma_indicators:
            ax_price.plot(df.loc[zoom[0]:zoom[1], ma], label=ma)
    else:
        ax_price.plot(df['close'], label='Close Price', color='blue')
        for ma in ma_indicators:
            ax_price.plot(df[ma], label=ma)

    ax_price.set_ylabel('Close Price / MA')
    ax_price.legend(loc='best')
    ax_price.grid(which='both')

    # Plot other indicators
    for indicator in other_indicators:
        if indicator == 'ATR':
            ax = ax_atr
        elif indicator == 'RSI':
            ax = ax_rsi
        else:
            continue

        if zoom:
            ax.plot(df.loc[zoom[0]:zoom[1], indicator], label=indicator)
        else:
            ax.plot(df[indicator], label=indicator)

        ax.set_ylabel(indicator)
        ax.legend(loc='best')
        ax.grid(which='both')

    # Plot actions (buy/sell points)
    if actions:
        if zoom:
            buy_points = df[(df['action'] == 'buy') & (df.index >= zoom[0]) & (df.index <= zoom[1])]
            sell_points = df[(df['action'] == 'sell') & (df.index >= zoom[0]) & (df.index <= zoom[1])]
        else:
            buy_points = df[df['action'] == 'buy']
            sell_points = df[df['action'] == 'sell']

        ax_price.scatter(buy_points.index, buy_points['close'], color='green', label='Buy', marker='^', alpha=1)
        ax_price.scatter(sell_points.index, sell_points['close'], color='red', label='Sell', marker='v', alpha=1)

    plt.xlabel('Index')
    plt.tight_layout()
    plt.show()


def resample_data(data, freq):
    """
    Resamples the given data to the desired frequency.

    Args:
    data (DataFrame): Data to resample.
    freq (str): Frequency to resample to only (5min, 15min, 1h, and 1d) values.

    Returns:
    DataFrame: Resampled data.
    """

    # Convert the 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Set 'date' as the index
    data.set_index('date', inplace=True)

    # Convert pandas DataFrame to Dask DataFrame
    ddata = dd.from_pandas(data, npartitions=2)

    # Resample each column to the desired frequency
    open_data = ddata['open'].resample(freq).first().compute()
    high_data = ddata['high'].resample(freq).max().compute()
    low_data = ddata['low'].resample(freq).min().compute()
    close_data = ddata['close'].resample(freq).last().compute()
    volume_data = ddata['volume'].resample(freq).sum().compute()

    # Combine the resampled data
    resampled_data = pd.concat([open_data, high_data, low_data, close_data, volume_data], axis=1)
    resampled_data.columns = ['open', 'high', 'low', 'close', 'volume']

    # Reset the index and drop rows with missing values
    resampled_data.reset_index(inplace=True)
    resampled_data.dropna(inplace=True)

    return resampled_data


def generate_actions(df, profit_ratio, horizon):
    """
    Generates actions (buy, sell, hold) based on the profit ratio and horizon.

    Args:
    df (DataFrame): Data to generate actions for.
    profit_ratio (float): Profit ratio to determine actions (ex: 5, 12, 20).
    horizon (int): Number of rows to look back for determining actions.

    Returns:
    DataFrame: Data with the new actions column.
    """

    # Create a new column for actions and initialize it to 'hold'
    df['action'] = 'hold'

    # Convert the 'close' column to a numpy array for efficient indexing
    close_prices = df['close'].values

    # For each row, check the 'horizon' number of rows in the past
    for i in range(len(df)):
        # Initialize the action as hold
        action = 'hold'
        current_price = close_prices[i]

        # Check every point within the horizon
        for j in range(1, horizon):
            # Get the price before 'j' number of rows
            past_price = close_prices[i - j] if i - j >= 0 else np.nan

            # Calculate the percentage change from the current price to the past price
            pct_change = (past_price - current_price) / past_price * 100

            future_price = close_prices[i + j] if i + j < len(close_prices) - 1 else np.nan
            # Calculate the percentage change from the current price to the future price
            fct_change = (future_price - current_price) / current_price * 100

            # If the percentage change is greater than or equal to 'profit_ratio', mark a 'buy' action
            if pct_change >= profit_ratio or fct_change >= profit_ratio:
                action = 'buy'
                break  # We can break the loop as soon as we find a 'buy' action
            # If the percentage change is less than or equal to '-profit_ratio', mark a 'sell' action
            elif pct_change <= -profit_ratio or fct_change <= -profit_ratio:
                action = 'sell'
                break  # We can break the loop as soon as we find a 'sell' action

        # Set the action for the current row
        df.loc[i, 'action'] = action

    return df


def calculate_atr(data, n=14):
    data['high_low'] = data['high'] - data['low']
    data['high_prev_close'] = abs(data['high'] - data['close'].shift())
    data['low_prev_close'] = abs(data['low'] - data['close'].shift())

    true_range = data[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    atr = true_range.rolling(n).mean()

    data.drop(['high_low', 'high_prev_close', 'low_prev_close'], axis=1, inplace=True)

    return atr


def calculate_moving_average(data, window):
    return data['close'].rolling(window).mean()


def calculate_rsi(data, n=14):
    delta = data['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    average_gain = up.rolling(n).mean()
    average_loss = abs(down.rolling(n).mean())

    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_indicators(data):
    # Ensure data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Add Average True Range (ATR)
    data['ATR'] = calculate_atr(data)

    # Add Moving Average (20, 100, 200)
    for window in [20, 100, 200]:
        data[f'MA_{window}'] = calculate_moving_average(data, window)

    # Add Relative Strength Index (RSI)
    data['RSI'] = calculate_rsi(data)

    return data


def preprocessing(filename, resample=None, profit_ratio=15, horizon=12,
                  indicators=None, plot_prices=False, plot_zoom=None, plot_actions=True):
    """
    Preprocesses the data, including resampling and generating actions column.

    Args:
    filename (str): Name of the file to load data from.
    resample (str): Frequency to resample to only (5min, 15min, 1h, and 1d) values. If None, will use 1min.
    profit_ratio (float): Profit ratio to determine actions.
    horizon (int): Number of price points from both past and future to look for determining actions (buy, sell, hold).
    plot_prices (bool): Whether to plot the prices.
    plot_zoom (list): Range of indices to zoom in on (ex: [200, 300])
    plot_actions (bool): Whether to plot the points of action.
    """

    start = timeit.default_timer()

    # Load the data
    df = pd.read_csv(filename)

    # Sort the data by date in ascending order
    df.sort_values('date', inplace=True, ascending=True)
    df.reset_index(drop=True, inplace=True)

    if resample:
        # Resample the data to different frequencies
        start_time = timeit.default_timer()
        df = resample_data(df, resample)
        end_time = timeit.default_timer()
        print(f"Resampling took {end_time - start_time:.2f} seconds.")

    # Generate actions column (sell, buy, hold)
    start_time = timeit.default_timer()
    df_actions = generate_actions(df, profit_ratio, horizon)
    end_time = timeit.default_timer()
    print(f"Adding actions column took {end_time - start_time:.2f} seconds.")

    # Add the indicators to the data
    data_with_indicators = add_indicators(df_actions)

    end = timeit.default_timer()
    print(f"preprocessing took {end - start:.2f} seconds.")

    # Plot prices and actions if required
    if plot_prices:
        plot(data_with_indicators, figsize=(14, 7), zoom=plot_zoom, actions=plot_actions,
             indicators=indicators)
