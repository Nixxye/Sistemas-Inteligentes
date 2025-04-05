import math
import random
import visualize_tsp
import matplotlib.pyplot as plt


class SimAnneal(object):
    def __init__(self, coords, T=100, alpha=0.95, stopping_T=1e-2, stopping_iter=100):
        self.coords = coords
        self.n = len(coords)
        self.T = T
        self.alpha = alpha
        self.stopping_T = stopping_T
        self.stopping_iter = stopping_iter
        self.best = None
        # importante salvar a temperatura e o distância percorrida em cada iteração

    def anneal(self):
        None
    
    def innit_solution(self):
        None
    
