"""Driver: 4x4 instance sweep, reusing the generic experiment."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import run_sk_sweep as exp

CFG = dict(exp.CONFIG,
           n=4, m=4,
           K_values=[1000, 5000, 20000],
           n_seeds=3,
           S_ref=5000,
           chunk=500,
           out_png="alg1_sk_sweep_4x4.png",
           out_csv="alg1_sk_sweep_4x4.csv")

if __name__ == "__main__":
    _, _, results = exp.run_experiment(CFG)
    exp.plot_results(CFG, results)
    exp.save_csv(CFG, results)
