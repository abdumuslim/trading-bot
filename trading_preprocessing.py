import timeit
import numpy as np
import pandas as pd
import dask.dataframe as dd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot(df, chart_type='line', figsize=(1400, 700), zoom=None, actions=None, indicators=None,
         show_candle_patterns=True, show_chart_patterns=True):
    """
    Plot data with different indicators, actions, and patterns on an interactive chart.

    Args:
        df (pd.DataFrame): Input data.
        chart_type (str): Type of chart to plot, either 'line' or 'candle'.
        figsize (tuple): Size of the figure.
        zoom (tuple): Range of data to zoom in on.
        actions (list): List of actions to annotate on the chart.
        indicators (list): List of indicators to plot ['MA_20', 'MA_50', 'MA_100', 'MA_200', 'RSI', 'ATR'].
        show_candle_patterns (bool): Whether to show candlestick patterns.
        show_chart_patterns (bool): Whether to show chart patterns.

    Raises:
        ValueError: If chart_type is not 'line' or 'candle'.
    """

    if chart_type not in ['line', 'candle']:
        raise ValueError('Invalid chart_type: Expected "line" or "candle"')

    if zoom:
        df = df.loc[zoom[0]:zoom[1], :]

    if indicators:
        # Create subplots, including a subplot for ATR at the top, RSI if needed and the main chart
        if 'RSI' in indicators and 'ATR' in indicators:
            indicators.remove('RSI')
            indicators.remove('ATR')
            fig = make_subplots(rows=4, cols=1, row_heights=[0.55, 0.15, 0.15, 0.15],
                                subplot_titles=('Prices', 'Volume', 'ATR', 'RSI'), vertical_spacing=0.08)

            fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], mode='lines', name='ATR', line=dict(width=1.5)), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(width=1.5)), row=4, col=1)
            fig.add_shape(go.layout.Shape(type="line", yref="y4", xref="x4",
                                          x0=min(df.index), y0=70, x1=max(df.index), y1=70,
                                          line=dict(color="Red", width=1, dash="dash")), row=4, col=1)
            fig.add_shape(go.layout.Shape(type="line", yref="y4", xref="x4",
                                          x0=min(df.index), y0=30, x1=max(df.index), y1=30,
                                          line=dict(color="Red", width=1, dash="dash")), row=4, col=1)

        elif 'ATR' in indicators:
            indicators.remove('ATR')
            fig = make_subplots(rows=3, cols=1, row_heights=[0.7, 0.15, 0.15],
                                subplot_titles=('Prices', 'Volume', 'ATR'), vertical_spacing=0.08)
            fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], mode='lines', name='ATR', line=dict(width=1.5)), row=3, col=1)

        elif 'RSI' in indicators:
            indicators.remove('RSI')
            fig = make_subplots(rows=3, cols=1, row_heights=[0.7, 0.15, 0.15],
                                subplot_titles=('Prices', 'Volume', 'RSI'), vertical_spacing=0.08)
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI',
                                     line=dict(width=1.5)), row=3, col=1)
            fig.add_shape(go.layout.Shape(type="line", yref="y3", xref="x3",
                                          x0=min(df.index), y0=70, x1=max(df.index), y1=70,
                                          line=dict(color="Red", width=1, dash="dash")), row=3, col=1)
            fig.add_shape(go.layout.Shape(type="line", yref="y3", xref="x3",
                                          x0=min(df.index), y0=30, x1=max(df.index), y1=30,
                                          line=dict(color="Red", width=1, dash="dash")), row=3, col=1)

        else:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2],
                                subplot_titles=('Prices', 'Volume'), vertical_spacing=0.08)
            # Add indicators
        for ind in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df[ind], mode='lines', name=ind, line=dict(width=1.5)),
                          row=1, col=1)

    else:
        fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2],
                            subplot_titles=('Prices', 'Volume'), vertical_spacing=0.08)

    if chart_type == 'candle':
        fig.add_trace(
            go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='OHLC'),
            row=1, col=1)

    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close',
                                 line=dict(width=2.5, color='black')), row=1, col=1)

    # Create Volume Bar chart
    colors = df.close.diff().apply(lambda x: 'green' if x >= 0 else 'red')
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker=dict(color=colors), name='Volume'), row=2, col=1)

    # Add actions
    if actions:
        if 'buy' in df['action'].values:
            fig.add_trace(
                go.Scatter(x=df[df['action'] == 'buy'].index, y=df[df['action'] == 'buy']['close'], mode='markers',
                           name='Buy', marker=dict(symbol='triangle-up', size=10, color='green', line_width=1)),
                row=1, col=1)
        if 'sell' in df['action'].values:
            fig.add_trace(
                go.Scatter(x=df[df['action'] == 'sell'].index, y=df[df['action'] == 'sell']['close'], mode='markers',
                           name='Sell', marker=dict(symbol='triangle-down', size=10, color='red', line_width=1)),
                row=1, col=1)

    # Add candlestick patterns
    if show_candle_patterns:
        patterns = {
            'Hammer': 'circle',
            'Bullish_Engulfing': 'star',
            'Bearish_Engulfing': 'x',
            'Close_Above_Candle': 'triangle-up',
            'Close_Below_Candle': 'triangle-down'
        }
        for pattern, symbol in patterns.items():
            if pattern in df.columns and df[pattern].any():
                fig.add_trace(
                    go.Scatter(x=df[df[pattern] == True].index, y=df[df[pattern] == True]['close'], mode='markers',
                               name=pattern, marker=dict(symbol=symbol, size=10, color='orange', line_width=1)),
                    row=1, col=1)

    if show_chart_patterns:
        # Add chart patterns
        chart_patterns = {
            'Double_Top': 'diamond',
            'Double_Bottom': 'diamond-open',
            'Flag_Pattern': 'x-thin',
            'Ascending_Wedge': 'y-up',
            'Descending_Wedge': 'y-down'
        }
        for pattern, symbol in chart_patterns.items():
            if pattern in df.columns and df[pattern].any():
                fig.add_trace(
                    go.Scatter(x=df[df[pattern] == True].index, y=df[df[pattern] == True]['close'], mode='markers',
                               name=pattern, marker=dict(symbol=symbol, size=10, color='purple', line_width=1)),
                    row=1, col=1)

    # Update layout
    fig.update_layout(height=figsize[1], width=figsize[0],
                      title_text="Interactive Chart with Indicators, Actions and Patterns",
                      xaxis_rangeslider_visible=False)
    fig.show()


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
    moving_avg = data['close'].rolling(window).mean()
    first_valid_index = moving_avg.first_valid_index()
    first_valid_value = moving_avg.loc[first_valid_index]
    moving_avg.fillna(first_valid_value, inplace=True)
    return moving_avg


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

    # Add Moving Average (20, 50, 100, 200)
    for window in [20, 50, 100, 200]:
        data[f'MA_{window}'] = calculate_moving_average(data, window)

    # Add Relative Strength Index (RSI)
    data['RSI'] = calculate_rsi(data)

    # Drop rows with NaN values
    data.dropna(inplace=True)

    # Reset index
    data.reset_index(drop=True, inplace=True)

    return data


def identify_candle_patterns(data):
    # Add empty columns for the patterns
    data['Hammer'] = False
    data['Bullish_Engulfing'] = False
    data['Bearish_Engulfing'] = False
    data['Close_Above_Candle'] = False
    data['Close_Below_Candle'] = False

    for i in range(1, len(data)):
        # Identify Hammer
        body = abs(data.loc[i, 'open'] - data.loc[i, 'close'])
        whole_candle = data.loc[i, 'high'] - data.loc[i, 'low']

        # Green Hammer
        if (data.loc[i, 'close'] > data.loc[i, 'open']) and \
                ((min(data.loc[i, 'open'], data.loc[i, 'close']) - data.loc[i, 'low']) >= 2 * body) and \
                (body / whole_candle >= 0) and (body / whole_candle <= 0.382):
            data.loc[i, 'Hammer'] = True
        # Red Hammer
        elif (data.loc[i, 'open'] > data.loc[i, 'close']) and \
                ((data.loc[i, 'high'] - min(data.loc[i, 'open'], data.loc[i, 'close'])) >= 2 * body) and \
                (body / whole_candle >= 0) and (body / whole_candle <= 0.382):
            data.loc[i, 'Hammer'] = True

        # Identify Bullish Engulfing
        if (data.loc[i - 1, 'open'] > data.loc[i - 1, 'close']) and \
                (data.loc[i, 'open'] < data.loc[i, 'close']) and \
                (data.loc[i - 1, 'open'] >= data.loc[i, 'close']) and \
                (data.loc[i - 1, 'close'] <= data.loc[i, 'open']):
            data.loc[i, 'Bullish_Engulfing'] = True

        # Identify Bearish Engulfing
        if (data.loc[i - 1, 'open'] < data.loc[i - 1, 'close']) and \
                (data.loc[i, 'open'] > data.loc[i, 'close']) and \
                (data.loc[i - 1, 'close'] >= data.loc[i, 'open']) and \
                (data.loc[i - 1, 'open'] <= data.loc[i, 'close']):
            data.loc[i, 'Bearish_Engulfing'] = True

        # Identify Close Above/Below Candle
        if data.loc[i, 'close'] > data.loc[i - 1, 'high']:
            data.loc[i, 'Close_Above_Candle'] = True
        elif data.loc[i, 'close'] < data.loc[i - 1, 'low']:
            data.loc[i, 'Close_Below_Candle'] = True

    return data


def identify_chart_patterns(data, tolerance=0.01):
    # Add empty columns for the patterns
    data['Double_Top'] = False
    data['Double_Bottom'] = False
    data['Flag_Pattern'] = False
    data['Ascending_Wedge'] = False
    data['Descending_Wedge'] = False

    # Identify Double Tops and Bottoms
    data['min'] = data['low'].rolling(window=3, center=True).min()
    data['max'] = data['high'].rolling(window=3, center=True).max()
    data['local_min'] = data['low'] == data['min']
    data['local_max'] = data['high'] == data['max']
    for i in range(3, len(data)):
        if data.loc[i, 'local_max'] and data.loc[i-3, 'local_max'] and \
           abs(data.loc[i, 'high'] - data.loc[i-3, 'high']) / data.loc[i-3, 'high'] < tolerance:
            data.loc[i, 'Double_Top'] = True
        if data.loc[i, 'local_min'] and data.loc[i-3, 'local_min'] and \
           abs(data.loc[i, 'low'] - data.loc[i-3, 'low']) / data.loc[i-3, 'low'] < tolerance:
            data.loc[i, 'Double_Bottom'] = True

    # Identify Flag Patterns
    data['volatility'] = data['high'] - data['low']
    data['above_MA20'] = data['close'] > data['MA_20']
    for i in range(1, len(data)):
        if data.loc[i, 'volatility'] < data.loc[i-1, 'volatility'] and data.loc[i, 'above_MA20']:
            data.loc[i, 'Flag_Pattern'] = True

    # Identify Ascending and Descending Wedges
    data['high_change'] = data['high'].pct_change()
    data['low_change'] = data['low'].pct_change()
    for i in range(1, len(data)):
        if data.loc[i, 'high_change'] > data.loc[i, 'low_change']:
            data.loc[i, 'Ascending_Wedge'] = True
        if data.loc[i, 'low_change'] > data.loc[i, 'high_change']:
            data.loc[i, 'Descending_Wedge'] = True

    # Drop auxiliary columns
    data.drop(['min', 'max', 'local_min', 'local_max', 'volatility', 'above_MA20', 'high_change', 'low_change'], axis=1, inplace=True)

    return data


def preprocessing(filename, resample=None, profit_ratio=15, horizon=12,
                  indicators=None, plot_prices=False, plot_zoom=None, plot_actions=True, chart_type='line',
                  candle_patterns=False, chart_patterns=True):

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
    df = generate_actions(df, profit_ratio, horizon)
    end_time = timeit.default_timer()
    print(f"Adding actions column took {end_time - start_time:.2f} seconds.")

    if indicators:
        # Add the indicators to the data
        start_time = timeit.default_timer()
        df = add_indicators(df)
        if 'all' in indicators:
            indicators = ['MA_20', 'MA_50', 'MA_100', 'MA_200', 'RSI', 'ATR']
        else:
            indicators = [ind for ind in indicators if ind in df.columns]

        end_time = timeit.default_timer()
        print(f"Adding indicators column took {end_time - start_time:.2f} seconds.")

    if candle_patterns:
        # Add the indicators to the data
        start_time = timeit.default_timer()
        df = identify_candle_patterns(df)
        end_time = timeit.default_timer()
        print(f"Adding candle patterns column took {end_time - start_time:.2f} seconds.")

    if chart_patterns:
        if indicators and 'MA_20' in indicators:
            # Add the indicators to the data
            start_time = timeit.default_timer()
            df = identify_chart_patterns(df)
            end_time = timeit.default_timer()
            print(f"Adding chart patterns column took {end_time - start_time:.2f} seconds.")
        else:
            print("Cannot make chart patterns without 'MA_20' indicator")

    end = timeit.default_timer()
    print(f"preprocessing took {end - start:.2f} seconds.")

    df = df[200:]
    # Plot prices and actions if required
    if plot_prices:
        plot(df, chart_type=chart_type, figsize=(1400, 700), zoom=plot_zoom, actions=plot_actions,
             indicators=indicators, show_candle_patterns=candle_patterns, show_chart_patterns=chart_patterns)
