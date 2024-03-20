import numpy as np
import pandas as pd

    
# https://www.codearmo.com/blog/sharpe-sortino-and-calmar-ratios-python
# https://blog.quantinsti.com/sharpe-ratio-applications-algorithmic-trading/
def compute_sharpe_ratio(excess_return: np.ndarray) -> float:
    """
    Compute the Sharpe Ratio for a given series of excess returns.

    Parameters:
    excess_return (np.ndarray): Array of excess returns for a portfolio.

    Returns:
    float: The Sharpe Ratio, calculated as the mean of the excess returns divided by their standard deviation.
    """
    # Calculating the Sharpe Ratio
    sr = np.mean(excess_return) / np.std(excess_return) * np.sqrt(len(excess_return))
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
    rolling_sr = rolling_mean / rolling_std * np.sqrt(window)
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
    sortino_ratio = np.mean(excess_return) / downside_std * np.sqrt(len(excess_return))
    return sortino_ratio

def compute_drawdown(excess_return: np.ndarray, max_drawdown: bool = True) -> np.ndarray:
    """
    Compute drawdown or maximum drawdown of a portfolio.

    Parameters:
    excess_return (np.ndarray): Array of portfolio values over time.
    max_drawdown (bool): If True, returns the maximum drawdown, otherwise returns the drawdown series.

    Returns:
    np.ndarray: Either the maximum drawdown value or an array of drawdown values over time.
    """
    
    if not isinstance(excess_return, np.ndarray):
        excess_return = np.array(excess_return)
    
    drawdown = np.zeros(len(excess_return))
    peak = excess_return[0]

    for i in range(1, len(excess_return)):
        if excess_return[i-1] == 0:
            continue
        peak = max(peak, excess_return[i])
        if peak == 0:
            peak = np.finfo(float).eps # Smallest positive float number
        
        drawdown[i] = (excess_return[i] - peak) / peak

    if max_drawdown:
        return np.min(drawdown)
    else:
        return drawdown
    
def compute_drawdown_pnl(pnl_series: np.ndarray) -> np.ndarray:
    """
    Calculate the drawdown of a PnL series.

    Parameters:
    pnl_series (np.ndarray): An array of cumulative profits and losses.

    Returns:
    np.ndarray: An array of drawdown values.
    """
    # Calculate the cumulative maximum of the PnL series up to each point
    running_max = np.maximum.accumulate(pnl_series)
    # Calculate drawdown
    drawdown = np.log(running_max - pnl_series)

    return drawdown    

def compute_max_drawdown(
        daily_excess_returns : np.ndarray, 
        cummulative_returns : np.ndarray 
        ) -> float:
    """
    Calculates the maximum drawdown of a strategy as a percentage

    Arguments:
    ----------
        daily_excess_returns    : {np.ndarray}
                                    > The daily excess returns of a strategy
                                    (already discounted by the risk-free rate)
        cummulative_returns     : {np.ndarray}
                                    > The cummulative returns of a strategy

    Returns:
    ----------
        max_drawdown            : {float}
                                    > The maximum drawdown of a strategy
                                    (already discounted by the risk-free rate)
    """
    # Compute the percentage returns
    pect_returns = daily_excess_returns / cummulative_returns
    
    # Compute the cumulative returns and the running maximum
    cum_returns = np.cumprod(1 + pect_returns)
    running_max = np.maximum.accumulate(cum_returns)
    
    # Compute the drawdowns and find the maximum drawdown
    drawdowns = (cum_returns / running_max) - 1
    max_drawdown = np.min(drawdowns)

    return max_drawdown



def compute_calmar_ratio(excess_return: np.ndarray) -> float:
    """
    Compute the Calmar Ratio for a given series of excess returns and portfolio values.

    Parameters:
    excess_return (np.ndarray): Array of excess returns for a portfolio.
    excess_return (np.ndarray): Array of portfolio values over time.

    Returns:
    float: The Calmar Ratio, calculated as the standard deviation of the excess returns divided by the maximum drawdown.
    """
    # Calculating the standard deviation of the excess returns
    
    mean_return = np.mean(excess_return)
    # Calculating the Calmar Ratio
    return mean_return / compute_drawdown(excess_return, True)


def compute_calmar_ratio_modify(daily_pnl: np.ndarray, cum_pnl: np.ndarray) -> float:
    """
    Compute the Calmar Ratio for a given series of excess returns and portfolio values.

    Parameters:
    excess_return (np.ndarray): Array of excess returns for a portfolio.
    excess_return (np.ndarray): Array of portfolio values over time.

    Returns:
    float: The Calmar Ratio, calculated as the standard deviation of the excess returns divided by the maximum drawdown.
    """
    # Calculating the standard deviation of the excess returns
    
    return_series = daily_pnl / cum_pnl
    mean_return = np.mean(return_series) * len(return_series)
    # Calculating the Calmar Ratio
    return mean_return / abs(compute_max_drawdown(daily_pnl, cum_pnl))

     
    