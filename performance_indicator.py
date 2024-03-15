import numpy as np
import pandas as pd

    

def compute_sharpe_ratio(excess_return: np.ndarray) -> float:
    """
    Compute the Sharpe Ratio for a given series of excess returns.

    Parameters:
    excess_return (np.ndarray): Array of excess returns for a portfolio.

    Returns:
    float: The Sharpe Ratio, calculated as the mean of the excess returns divided by their standard deviation.
    """
    # Calculating the Sharpe Ratio
    sr = np.mean(excess_return) / np.std(excess_return)
    return sr

def compute_rolling_sharpe_ratio(excess_return: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the rolling Sharpe Ratio for a given series of excess returns.

    Parameters:
    excess_return (np.ndarray): Array of excess returns for a portfolio.
    window (int): The rolling window size to calculate the Sharpe Ratio.

    Returns:
    np.ndarray: An array of rolling Sharpe Ratios.
    """
    # Converting to Pandas DataFrame for rolling window calculation
    temp_df = pd.DataFrame({'excess_return': excess_return})

    # Calculating rolling mean and standard deviation
    rolling_mean = temp_df['excess_return'].rolling(window = window).mean().to_numpy()
    rolling_std = temp_df['excess_return'].rolling(window = window).std().to_numpy()

    # Calculating the rolling Sharpe Ratio
    rolling_sr = rolling_mean / rolling_std
    return rolling_sr

def compute_sortino_ratio(excess_return: np.ndarray) -> float:
    """
    Compute the Sortino Ratio for a given series of excess returns.

    Parameters:
    excess_return (np.ndarray): Array of excess returns for a portfolio.

    Returns:
    float: The Sortino Ratio, calculated as the mean of the excess returns divided by the standard deviation of the negative excess returns.
    """
    # Isolating negative excess returns for downside deviation calculation
    negative_excess_returns = excess_return[excess_return < 0]

    # Calculating downside standard deviation
    downside_std = np.std(negative_excess_returns)

    # Calculating the Sortino Ratio
    sortino_ratio = np.mean(excess_return) / downside_std
    return sortino_ratio

def compute_drawdown(portfolio_value: np.ndarray, max_drawdown: bool = True) -> np.ndarray:
    """
    Compute drawdown or maximum drawdown of a portfolio.

    Parameters:
    portfolio_value (np.ndarray): Array of portfolio values over time.
    max_drawdown (bool): If True, returns the maximum drawdown, otherwise returns the drawdown series.

    Returns:
    np.ndarray: Either the maximum drawdown value or an array of drawdown values over time.
    """
    # Calculating the running maximum of the portfolio value
    running_max = np.maximum.accumulate(portfolio_value)

    # Calculating the drawdown
    drawdown = (running_max - portfolio_value) / running_max

    # Returning either the maximum drawdown or the drawdown series
    if not max_drawdown:
        return drawdown
    else:
        return np.max(drawdown)

def compute_calmar_ratio(excess_return: np.ndarray, portfolio_value: np.ndarray) -> float:
    """
    Compute the Calmar Ratio for a given series of excess returns and portfolio values.

    Parameters:
    excess_return (np.ndarray): Array of excess returns for a portfolio.
    portfolio_value (np.ndarray): Array of portfolio values over time.

    Returns:
    float: The Calmar Ratio, calculated as the standard deviation of the excess returns divided by the maximum drawdown.
    """
    # Calculating the standard deviation of the excess returns
    CR = np.mean(excess_return)

    # Calculating the Calmar Ratio
    return CR / compute_drawdown(portfolio_value, max_drawdown=True)

     
    