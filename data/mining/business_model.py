import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Dict

class DiskBusinessModel:
    def __init__(self,
                 income_per_month: float,
                 disk_cost: float,
                 failure_loss: float,
                 annual_failure_rate: float,
                 replacement_interval_years: float = None):
        self.R_month = income_per_month
        self.C_disk = disk_cost
        self.C_fail = failure_loss
        self.lambda_year = -np.log(1 - annual_failure_rate)
        self.T_rep = replacement_interval_years

    def simulate_profit(self,
                        num_disks: int,
                        num_years: int,
                        n_simulations: int = 10000,
                        random_seed: int = 42) -> np.ndarray:
        np.random.seed(random_seed)
        profits = []

        for _ in range(n_simulations):
            total_profit = 0.0
            for _ in range(num_disks):
                time_alive = 0.0
                profit = 0.0

                while time_alive < num_years:
                    if self.T_rep is not None:
                        time_to_replacement = self.T_rep
                        time_to_failure = np.random.exponential(1 / self.lambda_year)

                        if time_to_failure < time_to_replacement:
                            dt = min(time_to_failure, num_years - time_alive)
                            profit += self.R_month * 12 * dt
                            profit -= (self.C_disk + self.C_fail)
                            time_alive += dt
                        else:
                            dt = min(time_to_replacement, num_years - time_alive)
                            profit += self.R_month * 12 * dt
                            profit -= self.C_disk
                            time_alive += dt
                    else:
                        time_to_failure = np.random.exponential(1 / self.lambda_year)
                        dt = min(time_to_failure, num_years - time_alive)
                        profit += self.R_month * 12 * dt
                        if dt < (num_years - time_alive):
                            profit -= (self.C_disk + self.C_fail)
                        time_alive += dt
                total_profit += profit
            profits.append(total_profit)
        return np.array(profits)

    @staticmethod
    def compute_statistics(profits: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
        mean_profit = np.mean(profits)
        sorted_profits = np.sort(profits)
        cutoff_idx = int(len(sorted_profits) * alpha)
        cvar = np.mean(sorted_profits[:cutoff_idx])
        return {
            'mean_profit': mean_profit,
            'cvar': cvar,
            'profit_std': np.std(profits),
            'var_percentile': np.percentile(profits, alpha * 100)
        }

def load_disks_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def create_models_from_dataset(df: pd.DataFrame,
                                income_per_month: float,
                                failure_loss: float,
                                replacement_interval_years: float = None) -> Dict[Tuple[str, str], DiskBusinessModel]:
    models = {}
    grouped = df.groupby(['mfg', 'model'])
    for (mfg, model), group in grouped:
        if 'cost_estimate' not in group.columns or 'mtbf' not in group.columns:
            continue
        avg_cost = group['cost_estimate'].mean()
        avg_mtbf = group['mtbf'].mean()
        hours_per_year = 24 * 365
        annual_failure_rate = 1 - np.exp(-hours_per_year / (avg_mtbf * 1e3)) if avg_mtbf > 0 else 0.05

        models[(mfg, model)] = DiskBusinessModel(
            income_per_month=income_per_month,
            disk_cost=avg_cost,
            failure_loss=failure_loss,
            annual_failure_rate=annual_failure_rate,
            replacement_interval_years=replacement_interval_years
        )
    return models

def visualize_model_statistics(models: Dict[Tuple[str, str], DiskBusinessModel],
                                num_disks: int,
                                num_years: int,
                                n_simulations: int = 5000):
    stats_list = []

    for (mfg, model), business_model in tqdm(models.items()):
        profits = business_model.simulate_profit(num_disks=num_disks,
                                                 num_years=num_years,
                                                 n_simulations=n_simulations)
        stats = business_model.compute_statistics(profits)
        stats['model'] = f"{mfg} {model}"
        stats_list.append(stats)
        break

    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.sort_values(by='mean_profit', ascending=False)

    plt.figure(figsize=(14, 8))
    plt.scatter(stats_df['mean_profit'], stats_df['cvar'])

    for _, row in stats_df.iterrows():
        plt.text(row['mean_profit'], row['cvar'], row['model'], fontsize=6, alpha=0.7)

    plt.xlabel('Средняя прибыль')
    plt.ylabel('CVaR (Худший средний доход)')
    plt.title('Средняя прибыль vs CVaR по моделям дисков')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/business_model.png")

# Пример использования
if __name__ == "__main__":
    df = load_disks_dataset("output/train.csv")
    models = create_models_from_dataset(df,
                                        income_per_month=1000,
                                        failure_loss=5000,
                                        replacement_interval_years=5)

    visualize_model_statistics(models,
                                num_disks=100,
                                num_years=5,
                                n_simulations=5000)
