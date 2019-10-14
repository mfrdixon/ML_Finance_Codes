import numpy as np


class wiener_process_generator:
    
    @staticmethod
    def generate_dW(dt: float, n_paths: int, n_steps: int, seed=0) -> np.ndarray:
        np.random.seed(seed=0)
        return np.sqrt(dt) * np.random.normal(size=(n_paths, n_steps))
    
    @staticmethod
    def generate_W(dt: float, n_paths: int, n_steps: int, seed=0) -> np.ndarray:
        X = wiener_process_generator.generate_dW(dt, n_paths, n_steps, seed)
        X[:, 0] = 0.0 # initial values
        X = X.cumsum(axis=1)
        return X


class geometric_brownian_generator:
    
    @staticmethod
    def generate_S_and_dW(dt: float, n_paths: int, n_steps: int, mu: float, vol: float, init_val:float, seed=0):
        
        dW = wiener_process_generator.generate_dW(dt, n_paths, n_steps, seed)
        S = mu * dt + vol * dW
        for j in range(S.shape[1]):
            if j == 0:
                S[:, j] =  init_val
            else:
                S[:, j] = S[:, j-1] * (1.0+S[:, j])
        
        return S, dW
    
    @staticmethod
    def generate_S_and_W(dt: float, n_paths: int, n_steps: int, mu: float, vol: float, init_val:float, seed=0):
        
        W = wiener_process_generator.generate_W(dt, n_paths, n_steps, seed)
        mu = np.full_like(W, mu)
        vol = np.full_like(W, vol)
        t = np.zeros_like(W)
        for j in range(t.shape[1]): t[:, j] = j * dt
        
        S = init_val * np.exp(vol * W + (mu - vol**2 / 2) * t)
        
        return S, W


if __name__ == '__main__':
    pass