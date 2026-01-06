import torch
import click
import time
import yaml
import numpy as np

@click.command()
@click.option('--config', default='economy_config.yaml', help='Путь к YAML файлу конфигурации')
def run_holo_economy_yaml(config):
    # Загрузка параметров из YAML
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    
    m_p = params['market_params']
    p_p = params['product_params']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    click.secho(f"🚀 Запуск Holo-Quantum Engine (YAML config) на {device}...", fg='cyan')
    click.echo(f"Анализируем {m_p['scenarios_count']} сценариев...")

    # --- СЛОЙ 1: ГЕНЕРАЦИЯ ВЕРОЯТНОСТНЫХ ПОЛЕЙ (СИНГУМОВ) ---
    scenarios = m_p['scenarios_count']
    
    conv_field = p_p['conversion_min'] + torch.rand(scenarios, device=device) * (p_p['conversion_max'] - p_p['conversion_min'])
    cpc_field = torch.normal(m_p['cpc_avg'], m_p['cpc_stdev'], (scenarios,), device=device)
    price_field = torch.linspace(p_p['price_range_min'], p_p['price_range_max'], scenarios, device=device)

    # --- СЛОЙ 2: ДЕТАЛЬНАЯ ЮНИТ-ЭКОНОМИКА (ОПЕРАТОР ТРАНСФОРМАЦИИ) ---
    def calculate_full_model(prices, cpc_val, conv, budget):
        clicks = budget / cpc_val
        orders = clicks * conv
        cac = budget / (orders + 1e-6)
        
        # Переменные издержки зависят от цены и объема
        logistics = p_p['logistics_base'] + (prices * p_p['logistics_per_price'])
        unit_costs = p_p['base_cogs'] + logistics + (prices * p_p['tax_rate'])
        
        margin_per_unit = prices - unit_costs - cac
        total_profit = margin_per_unit * orders
        
        return total_profit, margin_per_unit

    # --- СЛОЙ 3: МГНОВЕННОЕ СХЛОПЫВАНИЕ ---
    start_time = time.time()
    with torch.inference_mode():
        total_profits, unit_margins = calculate_full_model(
            price_field, cpc_field, conv_field, m_p['budget']
        )
    
    best_idx = torch.argmax(total_profits)
    successful_scenarios = torch.sum(total_profits > 0).item()
    success_rate = (successful_scenarios / scenarios) * 100

    duration = time.time() - start_time

    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    click.secho(f"\n✅ Расчет завершен за {duration:.4f} сек.", fg='green')
    click.echo("-" * 40)
    click.echo(f"Оптимальная цена: {price_field[best_idx]:.2f} руб.")
    click.echo(f"Макс. потенциальная прибыль: {total_profits[best_idx]:.2f} руб. (при бюджете {m_p['budget']})")
    click.echo(f"Маржинальность в пике: {unit_margins[best_idx]:.2f} руб./ед.")
    
    color = 'green' if success_rate > 70 else 'yellow' if success_rate > 40 else 'red'
    click.secho(f"Устойчивость модели (вероятность прибыли): {success_rate:.2f}%", fg=color)
    click.echo("-" * 40)

if __name__ == '__main__':
    run_holo_economy_yaml()
