
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_position_value(df: pd.DataFrame, 
                           signal: np.array, 
                           initial_capital: int, 
                           max_leverage: int, 
                           reduced_leverage_shorting: bool = False,
                           hold_at_signal_0: bool = False) -> dict:

    N = len(df)
    
    price = df['Close'].to_numpy()
    daily_EFFR = df['Daily_EFFR'].to_numpy()
    daily_excess_return = df['Daily_excess_return'].to_numpy()

    unit = np.zeros(N)
    M = np.zeros(N) # Margin series

    # V: total value of the holdings
    # Vcap: unused capital (earn risk-free date)
    # Vtot: total value (V + Vcap)

    V, Vcap = np.zeros(N), np.zeros(N)
    dV, dVcap, dVtot = np.zeros(N), np.zeros(N), np.zeros(N)

    # 95% of money is used to trade and 5% of money is left in Vcap for emergency
    V[0] = initial_capital * 0.95
    Vcap[0] = initial_capital * 0.05
    Vtot = V + Vcap

    # theta: dollar value of ETF held at time t (How much money are you invested into the ETF)
    theta = np.zeros(N)    
    
    for t, signal in enumerate(signal):
        if t == 0:
            dVcap[t] = Vcap[t] * daily_EFFR[t]
            Vtot[t] = Vcap[t] + dVcap[t] + V[t]
            continue
        
        if signal > 0:
            leverage = max_leverage
        else:
            if reduced_leverage_shorting:
                leverage = max_leverage/2
            else:
                leverage = max_leverage
        
        if hold_at_signal_0 and signal == 0: # Hold
            unit[t] = unit[t-1]
            theta[t] = unit[t] * price[t]
            
        else:
            theta[t] = V[t-1] * leverage * signal
            # theta[t] = Vtot[t-1] * leverage * signal
            unit[t] = theta[t] / price[t]

        dV[t] = daily_excess_return[t] * theta[t]
        V[t] = V[t-1] + dV[t]
        
        # If the position value is greater than the bound, which is +- Vtot * L
        if np.abs(theta[t]) > V[t] * leverage:
            # Vcap[t] += theta[t] - V[t] * leverage
            # If the difference is +ve, you need to sell shares to fund Vcap.
            # If the difference id -ve, you need to buy shares to fund investment.
            # unit[t] -= (theta[t] - (V[t] * leverage * np.sign(theta[t])))/price[t]
            
            # New version
            diffs = np.abs(theta[t]) - V[t] * leverage
            unit[t] -= np.sign(theta[t]) * diffs / price[t]
            theta[t] -= np.sign(theta[t]) * diffs
            Vcap[t] += np.sign(theta[t]) * diffs
            
            # Recalculate dV and V after adjustment
            dV[t] = daily_excess_return[t] * theta[t]
            V[t] = V[t-1] + dV[t]

        M[t] = np.abs(theta[t])/leverage # Total margin used

        dVcap[t] = (Vtot[t-1] - M[t]) * daily_EFFR[t] 
        
        # Vtot[t-1] - M[t] refers to the total amount of money not being used to invest in ETF, this value should not be less than 0
        # if dVcap[t] < 0:
        #     dVcap[t] = 0
        Vcap[t] = Vcap[t-1] + dVcap[t]
        dVtot[t] = dV[t] + dVcap[t]
        Vtot[t] = Vtot[t-1] + dVtot[t]
        
        
    return {'Vtot': Vtot,'Vcap': Vcap, 'V': V,
            'dVtot': dVtot,'dVcap': dVcap, 'dV': dV,
            'theta': theta, 'M': M}
    
    

def calculate_turnover(theta: np.ndarray, price: np.array, mode = 'dollar'):
    if mode == 'dollar':
        return np.sum(np.abs(np.diff(theta)))
    elif mode == 'unit':
        return np.sum(np.abs(np.diff(theta/price)))
    else:
        raise ValueError(f'Mode {mode} is not found.')
    
    
def calculate_rolling_turnover(theta: np.ndarray, price: np.array, window_size=7, mode='dollar'):
    if mode == 'dollar':
        turnover = [np.sum(np.abs(np.diff(theta[i:i+window_size]))) for i in range(len(theta) - window_size + 1)]
    elif mode == 'unit':
        turnover = [np.sum(np.abs(np.diff(theta[i:i+window_size] / price[i:i+window_size]))) for i in range(len(theta) - window_size + 1)]
    else:
        raise ValueError(f'Mode {mode} is not found.')
    
    turnover_pad = [0] * (window_size - 1) + turnover
    return np.array(turnover_pad)
    
    
def calculate_cummulative_turnover(theta: np.ndarray, price: np.array, mode='dollar'):
    if mode == 'dollar':
        turnover = [np.sum(np.abs(np.diff(theta[0:i]))) for i in range(len(theta)+ 1)]
    elif mode == 'unit':
        turnover = [np.sum(np.abs(np.diff(theta[0:i] / price[0:i]))) for i in range(len(theta)+ 1)]
    else:
        raise ValueError(f'Mode {mode} is not found.')
    
    return np.array(turnover)

def calculate_PnL(Vtot: np.ndarray):
    return np.diff(Vtot)
    

# Reference version (NOT USED)
def evaluate_strategy(df, signals, init_capital=200000, max_leverage=10, control_leverage=False, hold=False):
    trade_days = len(df)

    prices = df['Close'].to_numpy()
    effrs = df['Daily EFFR'].to_numpy()
    excess_returns = df['Excess return'].to_numpy()

    units = np.zeros(trade_days)
    theta = np.zeros(trade_days)
    margins = np.zeros(trade_days)

    dV = np.zeros(trade_days)
    dV_cap = np.zeros(trade_days)
    dV_total = np.zeros(trade_days)

    V = np.zeros(trade_days)
    V[0] = init_capital
    V_cap = np.zeros(trade_days)
    V_total = V + V_cap

    for t, signal in enumerate(signals):
        # First day, no trade
        if t == 0:
            V_cap[t] = V_total[t] - margins[t]
            dV_cap[t] = V_cap[t] * effrs[t]
            dV_total[t] = dV_cap[t] + dV[t]
            continue

        V_total[t] = V_total[t-1] + dV_total[t-1]
        V[t] = V[t-1] + dV[t-1]

        if signal == 1:
            leverage = max_leverage
        else:
            leverage = max_leverage / 2 if control_leverage else max_leverage

        if signal == 0 and hold:
            units[t] = units[t-1]
            theta[t] = units[t] * prices[t]
        else:
            theta[t] = signal * V[t] * leverage
            units[t] = theta[t] / prices[t]

        # Adjust the theta according to the trade value before calculating the pnl
        # If theta > V * leverage, adjust the units held, cut the excess value, save it to capital,
        # If theta < -V * leverage, adjust the units held, cut the excess value from capital.
        # If not enough money in capital, out of trading.
        if np.abs(theta[t]) > V[t] * leverage:
            diffs = np.abs(theta[t]) - V[t] * leverage
            units[t] -= np.sign(theta[t]) * diffs / prices[t]
            theta[t] -= np.sign(theta[t]) * diffs
            V_cap[t] += np.sign(theta[t]) * diffs

        # Calculate the margin used and get the final capital value
        margins[t] = np.abs(theta[t]) / leverage
        V_cap[t] += V_total[t] - margins[t]

        # Calculate the daily pnl
        dV[t] = theta[t] * excess_returns[t]
        dV_cap[t] = V_cap[t] * effrs[t]
        dV_total[t] = dV[t] + dV_cap[t]

    wrapped_data = {
        'units_held': units,
        'margins': margins,
        'trade_pnl': dV,
        'cap_pnl': dV_cap,
        'total_pnl': dV_total,
        'theta': theta,
        'trade_vals': V,
        'cap_vals': V_cap,
        'total_vals': V_total
    }

    return wrapped_data