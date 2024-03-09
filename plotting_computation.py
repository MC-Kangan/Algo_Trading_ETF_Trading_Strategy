

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_position(date, theta, V, leverage):
    
    fig, ax = plt.subplots(1, 1)

    # plot position on training set
    ax.plot(date, theta, color = 'black', lw = 1, alpha = 1, label = r'$\theta_t$')
    ax.fill_between(date, - leverage * V, leverage * V, color = 'red', alpha = 0.2, label = r'$[-V \cdot L, V \cdot L]$')
    # ax.set_title('Training Set')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position in dollars (USD)')
    ax.legend()
    ax.grid(True)
    fig.show()
    

def plot_PnL(date, V, Vcap, Vtot):
    
    fig, ax = plt.subplots(2, 3, figsize = (15, 9))

    # plot position on training set
    ax[0, 0].plot(date[1:], np.diff(V), color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t$')
    # ax[0, 0].set_title('Training Set')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Profit and Loss in asset (USD)')
    ax[0, 0].grid(True)
    
    ax[0, 1].plot(date[1:], np.diff(Vcap), color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t^cap$')
    # ax[0, 1].set_title('Training Set')
    ax[0, 1].set_xlabel('Time')
    ax[0, 1].set_ylabel('Profit and Loss in unused capital (USD)')
    ax[0, 1].grid(True)
    
    ax[0, 2].plot(date[1:], np.diff(Vtot), color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t^total$')
    # ax[0, 2].set_title('Training Set')
    ax[0, 2].set_xlabel('Time')
    ax[0, 2].set_ylabel('Profit and Loss in total capital (USD)')
    ax[0, 2].grid(True)
    
    ax[1, 0].plot(date[1:], np.cumsum(np.diff(V)), color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t$')
    # ax[1, 0].set_title('Training Set')
    ax[1, 0].set_xlabel('Time')
    ax[1, 0].set_ylabel('Cummulative profit and Loss in asset (USD)')
    ax[1, 0].grid(True)
    
    ax[1, 1].plot(date[1:], np.cumsum(np.diff(Vcap)), color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t^cap$')
    # ax[1, 1].set_title('Training Set')
    ax[1, 1].set_xlabel('Time')
    ax[1, 1].set_ylabel('Cummulative profit and Loss in unused capital (USD)')
    ax[1, 1].grid(True)
    
    ax[1, 2].plot(date[1:], np.cumsum(np.diff(Vtot)), color = 'black', lw = 1, alpha = 1, label = r'$\deltaV_t^total$')
    # ax[1, 2].set_title('Training Set')
    ax[1, 2].set_xlabel('Time')
    ax[1, 2].set_ylabel('Cummulative profit and Loss in total capital (USD)')
    ax[1, 2].grid(True)
    fig.tight_layout()
    fig.show()