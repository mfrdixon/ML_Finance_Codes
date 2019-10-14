import numpy as np
import scipy.stats as stats


class call_option:
    
    @staticmethod
    def calc_deltas(S, K, ir, vol, T):
        
        dt = T / (S.shape[1]-1)
        
        # Convert to np.ndarray
        K = np.full_like(S, K)
        ir = np.full_like(S, ir)
        vol = np.full_like(S, vol)
        
        T = np.full_like(S, T)
        t = np.zeros_like(S)
        for j in range(t.shape[1]): t[:, j] = j*dt
        tau = T - t
        deltas = np.zeros_like(S)
        
        # Calc prices before expiry
        dplus = call_option.__dplus(S[:, :-1], K[:, :-1], ir[:, :-1], vol[:, :-1], tau[:, :-1])
        deltas[:, :-1] = stats.norm.cdf(dplus)
        
        # Calc prices at expiry
        deltas[:, -1] = np.where(S[:, -1] > K[:, -1], 1.0, 0.0)
        
        return deltas
    
    @staticmethod
    def calc_prices(S, K, ir, vol, T):
        
        dt = T / (S.shape[1]-1)
        
        # Convert to np.ndarray
        K = np.full_like(S, K)
        ir = np.full_like(S, ir)
        vol = np.full_like(S, vol)
        
        T = np.full_like(S, T)
        t = np.zeros_like(S)
        for j in range(t.shape[1]): t[:, j] = j*dt
        tau = T - t
        prices = np.zeros_like(S)
        
        # Calc prices before expiry
        dplus = call_option.__dplus(S[:, :-1], K[:, :-1], ir[:, :-1], vol[:, :-1], tau[:, :-1])
        dminus = call_option.__dminus(S[:, :-1], K[:, :-1], ir[:, :-1], vol[:, :-1], tau[:, :-1])
        df = np.exp(-ir[:, :-1] * tau[:, :-1])
        prices[:, :-1] = S[:, :-1] * stats.norm.cdf(dplus) - K[:, :-1] * df * stats.norm.cdf(dminus)
        
        # Calc prices at expiry
        prices[:, -1] = np.where(S[:, -1] > K[:, -1], S[:, -1] - K[:, -1], 0.0)
        
        return prices
    
    @staticmethod
    def calc_delta(S, K, ir, vol, T, t):
        tau = T - t
        if tau <= 0:
            return 1 if S > K else 0
        else:
            dplus = call_option.__dplus(S, K, ir, vol, tau)
            return stats.norm.cdf(dplus) 
    
    @staticmethod
    def calc_price(S, K, ir, vol, T, t):
        
        tau = T - t
        if tau <= 0: return max(S - K, 0)
        
        dplus = call_option.__dplus(S, K, ir, vol, tau)
        dminus = call_option.__dminus(S, K, ir, vol, tau)
        df = np.exp(-ir*tau)
        
        return S * stats.norm.cdf(dplus) - K * df * stats.norm.cdf(dminus) 
    
    @staticmethod
    def __dplus(S, K, ir, vol, tau):
        x = np.log(S/K) + (ir + vol**2 / 2) * tau
        return x / (vol * np.sqrt(tau))
    
    @staticmethod
    def __dminus(S, K, ir, vol, tau):        
        x = np.log(S/K) + (ir - vol**2 / 2) * tau
        return x / (vol * np.sqrt(tau))


if __name__ == '__main__':
    pass