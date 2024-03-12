
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

    Vcap[0] = initial_capital
    Vtot = V + Vcap

    # theta: dollar value of ETF held at time t (How much money are you invested into the ETF)
    theta = np.zeros(N)    
    
    for t, signal in enumerate(signal):
        if t == 0:
            dVcap[t] = Vcap[t] * daily_EFFR[t]
            Vtot[t] = Vcap[t] + dVcap[t]
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
            theta[t] = Vtot[t-1] * leverage * signal
            unit[t] = theta[t] / price[t]

        dV[t] = daily_excess_return[t] * theta[t]
        V[t] = V[t-1] + dV[t]

        M[t] = np.abs(theta[t])/leverage # Total margin used

        dVcap[t] = (Vtot[t-1] - M[t]) * daily_EFFR[t] 
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
    

def calculate_PnL(Vtot: np.ndarray):
    return np.diff(Vtot)
    


    