

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_signals(strategy_data, signal):
    # Plot the buy and sell signals on top of the closing price
    
    plt.figure(figsize=(14, 4))
    plt.plot(strategy_data['Date'], strategy_data['Close'], label='Close Price',  color = 'black', lw = 1,  alpha=0.5)
    
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    
    # Find the price and date when buying and selling
    buy_price = np.array(strategy_data['Close'])[np.where(signal == 1)]
    buy_price_time = np.array(strategy_data['Date'])[np.where(signal == 1)]
    sell_price = np.array(strategy_data['Close'])[np.where(signal == -1)]
    sell_price_time = np.array(strategy_data['Date'])[np.where(signal == -1)]
    
    plt.scatter(buy_price_time, buy_price, label='Buy Signal', marker='^', alpha=1, color='green')
    plt.scatter(sell_price_time, sell_price, label='Sell Signal', marker='v', alpha=1, color='red')
    plt.title('Buy & Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.legend(loc='upper left')
    
def plot_signals_compare(train_data, train_signal, test_data, test_signal):
    # Plot the buy and sell signals on top of the closing price
    
    fig, ax = plt.subplots(1, 2, figsize = (14, 4))
    if not isinstance(train_signal, np.ndarray):
        train_signal = np.array(train_signal)
    if not isinstance(test_signal, np.ndarray):
        test_signal = np.array(test_signal)
    
    ax[0].plot(train_data['Date'], train_data['Close'], label='Close Price',  color = 'black', lw = 1,  alpha=0.5)
    ax[1].plot(test_data['Date'], test_data['Close'], label='Close Price',  color = 'black', lw = 1,  alpha=0.5)
    
    def get_buy_sell_price(data, signal):
        # Find the price and date when buying and selling
        buy_price = np.array(data['Close'])[np.where(signal == 1)]
        buy_price_time = np.array(data['Date'])[np.where(signal == 1)]
        sell_price = np.array(data['Close'])[np.where(signal == -1)]
        sell_price_time = np.array(data['Date'])[np.where(signal == -1)]
        
        return buy_price, buy_price_time, sell_price, sell_price_time         

    buy_price, buy_price_time, sell_price, sell_price_time = get_buy_sell_price(train_data, train_signal)
    ax[0].scatter(buy_price_time, buy_price, label='Buy Signal', marker='^', alpha=1, color='green')
    ax[0].scatter(sell_price_time, sell_price, label='Sell Signal', marker='v', alpha=1, color='red')
    ax[0].set_title('Buy & Sell Signals - Training set')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Close Price USD ($)')
    ax[0].legend(loc='upper left')
    ax[0].tick_params(axis = 'x', labelrotation=45)
    
    buy_price, buy_price_time, sell_price, sell_price_time = get_buy_sell_price(test_data, test_signal)
    ax[1].scatter(buy_price_time, buy_price, label='Buy Signal', marker='^', alpha=1, color='green')
    ax[1].scatter(sell_price_time, sell_price, label='Sell Signal', marker='v', alpha=1, color='red')
    ax[1].set_title('Buy & Sell Signals - Testing set')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Close Price USD ($)')
    ax[1].tick_params(axis = 'x', labelrotation=45)
    # ax[1].legend(loc='upper left')
    
    return fig, ax
    

def plot_position(date, theta, V, leverage):
    
    fig, ax = plt.subplots(1, 1, figsize = (14, 4))

    # plot position on training set
    ax.plot(date, theta, color = 'black', lw = 1, alpha = 1, label = r'$\theta_t$')
    ax.fill_between(date, - leverage * V, leverage * V, color = 'red', alpha = 0.2, label = r'$[-V \cdot L, V \cdot L]$')
    # ax.set_title('Training Set')
    ax.set_xlabel('Date')
    ax.set_ylabel('Position in dollars (USD)')
    ax.legend(loc='upper left')
    ax.grid(True)
    fig.show()
    
    
def plot_position_compare(train_set_date, train_theta, train_V, test_set_date, test_theta, test_V, leverage):
    
    fig, ax = plt.subplots(1, 2, figsize = (14, 4))

    # plot position on training set
    ax[0].plot(train_set_date, train_theta, color = 'black', lw = 1, alpha = 1, label = r'$\theta_t$')
    ax[0].fill_between(train_set_date, - leverage * train_V, leverage * train_V, color = 'red', alpha = 0.2, label = r'$[-V \cdot L, V \cdot L]$')
    ax[0].set_title('Training set')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Position in dollars (USD)')
    ax[0].legend(loc='upper left')
    ax[0].grid(True)
    ax[0].tick_params(axis = 'x', labelrotation=45)
    
    # plot position on testing set
    ax[1].plot(test_set_date, test_theta, color = 'black', lw = 1, alpha = 1, label = r'$\theta_t$')
    ax[1].fill_between(test_set_date, - leverage * test_V, leverage * test_V, color = 'red', alpha = 0.2, label = r'$[-V \cdot L, V \cdot L]$')
    ax[1].set_title('Testing set')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Position in dollars (USD)')
    ax[1].legend(loc='upper left')
    ax[1].grid(True)
    ax[1].tick_params(axis = 'x', labelrotation=45)
    fig.show()
    
    return fig, ax
    

def plot_PnL(date, dV, dVcap, dVtot):

    fig, ax = plt.subplots(2, 3, figsize = (14, 6))
   
    # plot position on training set
    ax[0, 0].plot(date, dV, color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t$')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel(r'$\Delta V_t$')
    ax[0, 0].set_title('PnL in asset')
    ax[0, 0].grid(True)
    ax[0, 0].tick_params(axis = 'x',labelrotation=45)
    
    ax[0, 1].plot(date, dVcap, color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t^cap$')
    ax[0, 1].set_title('PnL in unused capital')
    ax[0, 1].set_xlabel('Time')
    ax[0, 1].set_ylabel(r'$\Delta V_t^{cap}$')
    ax[0, 1].grid(True)
    ax[0, 1].tick_params(axis = 'x',labelrotation=45)
    
    ax[0, 2].plot(date, dVtot, color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t^total$')
    ax[0, 2].set_title('PnL in total capital')
    ax[0, 2].set_xlabel('Time')
    ax[0, 2].set_ylabel(r'$\Delta V_t^{total}$')
    ax[0, 2].grid(True)
    ax[0, 2].tick_params(axis = 'x',labelrotation=45)
    
    ax[1, 0].plot(date, np.cumsum(dV), color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t$')
    ax[1, 0].set_title('Cumulative PnL in asset')
    ax[1, 0].set_xlabel('Time')
    ax[1, 0].set_ylabel(r'Cumulative $\Delta V_t$')
    ax[1, 0].grid(True)
    ax[1, 0].tick_params(axis = 'x',labelrotation=45)
    
    ax[1, 1].plot(date, np.cumsum(dVcap), color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t^cap$')
    ax[1, 1].set_title('Cumulative PnL in unused capital')
    ax[1, 1].set_xlabel('Time')
    ax[1, 1].set_ylabel(r'Cumulative $\Delta V_t^{cap}$')
    ax[1, 1].grid(True)
    ax[1, 1].tick_params(axis = 'x',labelrotation=45)
    
    ax[1, 2].plot(date, np.cumsum(dVtot), color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t^total$')
    ax[1, 2].set_title('Cumulative PnL in total capital')
    ax[1, 2].set_xlabel('Time')
    ax[1, 2].set_ylabel(r'Cumulative $\Delta V_t^{total}$')
    ax[1, 2].grid(True)
    ax[1, 2].tick_params(axis = 'x', labelrotation=45)

    fig.tight_layout()
    fig.show()
    
    
def plot_strategy(df: pd.DataFrame, result_dict: dict, signal: np.array, leverage: int):
    
    plot_signals(df, signal)
    plot_position(df['Date'], result_dict['theta'], result_dict['V'], leverage) 
    plot_PnL(df['Date'], result_dict['dV'], result_dict['dVcap'], result_dict['dVtot'])



def plot_turnover(df, turnover_dollar, turnover_unit, mode = 'cummulative'):
    fig, ax = plt.subplots(1, 2, figsize = (12, 3))
    ax[0].plot(df['Date'], turnover_dollar, lw = 1, color = 'black', label = 'Turnover in dollars')
    ax[0].tick_params(axis = 'x', labelrotation=45)
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel(r"$Turnover_{dollars}$")
    ax[0].set_title(f'{mode} turnover in dollars')
    # ax1 = ax.twinx()
    ax[1].plot(df['Date'], turnover_unit, lw = 1, color = 'black', label = 'Turnover in dollars')
    ax[1].tick_params(axis = 'x', labelrotation=45)
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel(r"$Turnover_{units}$")
    ax[1].set_title(f'{mode} turnover in units')
    # ax.legend(loc = 'upper left')
    fig.tight_layout()
    fig.show()
    return fig, ax
    

def calculate_cummulative_turnover(theta: np.ndarray, price: np.array, mode='dollar'):
    if mode == 'dollar':
        turnover = [np.sum(np.abs(np.diff(theta[0:i]))) for i in range(len(theta))]
    elif mode == 'unit':
        turnover = [np.sum(np.abs(np.diff(theta[0:i] / price[0:i]))) for i in range(len(theta))]
    else:
        raise ValueError(f'Mode {mode} is not found.')
    
    return np.array(turnover)