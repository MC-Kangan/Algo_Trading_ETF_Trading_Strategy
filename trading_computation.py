
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_position_value(df: pd.DataFrame, signal: np.array, initial_capital: int, leverage: int) -> dict:

    N = len(df)
    
    price = df['Close'].to_numpy()
    daily_EFFR = df['Daily_EFFR'].to_numpy()
    daily_excess_return = df['Daily_excess_return'].to_numpy()

    unit = np.zeros(N)

    L = leverage # Leverage
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

    indice = []
    for t, signal in enumerate(signal, 1):
        
        theta[t] = Vtot[t-1] * L * signal
        # Check if result is NaN
        if np.isnan(theta[t]):
            indice.append(t)

        unit[t] = theta[t] / price[t]
        dV[t] = daily_excess_return[t] * theta[t]
        V[t] = V[t-1] + dV[t]

        M[t] = np.abs(theta[t])/L # Total margin used

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
    


    