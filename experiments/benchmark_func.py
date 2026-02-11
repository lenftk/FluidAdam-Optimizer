import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fluid_adam import FluidAdam

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def sphere_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x**2)

def rosenbrock_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)

def rastrigin_function(x: torch.Tensor) -> torch.Tensor:
    A = 10.0
    return A * x.shape[0] + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x))

def run_benchmark(problems: dict, optimizers_config: dict, dims: int, iterations: int):
    results = {}
    for prob_name, func in problems.items():
        print(f"\n{'='*20} Benchmarking on {prob_name} (D={dims}) {'='*20}")
        results[prob_name] = {}
        initial_params = torch.randn(dims) * 2.0
        
        for opt_name, opt_config in optimizers_config.items():
            print(f"--- Running {opt_name} ---")
            loss_history, start_time = [], time.time()
            
            params = torch.nn.Parameter(initial_params.clone())
            optimizer = opt_config["class"]([params], **opt_config["args"])
            
            for i in range(iterations):
                optimizer.zero_grad()
                loss = func(params)
                if torch.isnan(loss):
                    print(f"Loss is NaN at iteration {i+1}. Stopping.")
                    loss_history.extend([np.nan] * (iterations - len(loss_history)))
                    break
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
                
                if (i + 1) % (max(1, iterations // 10)) == 0:
                    print(f"Iter {i+1}/{iterations}, Loss: {loss.item():.6f}")
            
            end_time = time.time()
            results[prob_name][opt_name] = {"history": loss_history, "time": end_time - start_time}
            print(f"{opt_name} finished in {end_time - start_time:.2f}s. Final loss: {loss_history[-1]:.6f}")
            
    return results

def plot_results(results: dict, dims: int, iterations: int):
    num_problems = len(results)
    fig, axes = plt.subplots(1, num_problems, figsize=(6 * num_problems, 5), squeeze=False)
    for i, (prob_name, opt_results) in enumerate(results.items()):
        ax = axes[0, i]
        for opt_name, data in opt_results.items():
            ax.plot(np.arange(len(data["history"])), data["history"], 
                    label=f"{opt_name} ({data['time']:.2f}s)")
        
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.set_title(f"'{prob_name}' Function (D={dims})")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss (Log Scale)")
        ax.legend()
        ax.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DIMENSIONS = 100000
    ITERATIONS = 2000
    LEARNING_RATE = 1e-3
    
    problems_to_test = {
        "Sphere": sphere_function,
        "Rosenbrock": rosenbrock_function,
        "Rastrigin": rastrigin_function,
    }
    
    optimizers_to_test = {
        "Adam": {
            "class": torch.optim.Adam,
            "args": {"lr": LEARNING_RATE, "betas": (0.9, 0.999)}
        },
        "FluidAdam_V4": {
            "class": FluidAdam,
            "args": {"lr": LEARNING_RATE, "betas": (0.9, 0.999), "nu": 0.7}
        }
    }
    
    benchmark_results = run_benchmark(problems_to_test, optimizers_to_test, DIMENSIONS, ITERATIONS)
    plot_results(benchmark_results, DIMENSIONS, ITERATIONS)