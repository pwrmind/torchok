import torch
import click
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

@click.command()
@click.option('--config', required=True, help='Путь к YAML файлу (например, detailing.yaml)')
@click.option('--plot', is_flag=True, help='Сгенерировать график ландшафта прибыли')
def run_holo_economy(config, plot):
    # Извлекаем имя конфига для именования файлов (напр. "detailing" из "detailing.yaml")
    config_path = Path(config)
    project_name = config_path.stem 

    with open(config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    m_p, p_p = params['market_params'], params['product_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenarios = int(m_p['scenarios_count'])

    # --- Универсальная генерация полей ---
    if 'conversion_to_lead_min' in p_p:
        c_l = torch.distributions.Uniform(p_p['conversion_to_lead_min'], p_p['conversion_to_lead_max']).sample((scenarios,)).to(device)
        c_s = torch.distributions.Uniform(p_p['conversion_to_sale_min'], p_p['conversion_to_sale_max']).sample((scenarios,)).to(device)
        conv_field = c_l * c_s
    else:
        conv_field = torch.distributions.Uniform(p_p['conversion_min'], p_p['conversion_max']).sample((scenarios,)).to(device)

    cpc_field = torch.normal(m_p['cpc_avg'], m_p['cpc_stdev'], size=(scenarios,), device=device).clamp(min=1.0)
    price_field = torch.distributions.Uniform(p_p['price_range_min'], p_p['price_range_max']).sample((scenarios,)).to(device)

    # --- Универсальный расчет ---
    with torch.inference_mode():
        sales = (m_p['budget'] / cpc_field) * conv_field
        unit_margin = price_field - (price_field * p_p['tax_rate']) - p_p['base_cogs']
        profits = (sales * unit_margin) - m_p['budget']

    # --- Аналитика ---
    best_idx = torch.argmax(profits)
    success_rate = (torch.sum(profits > 0).item() / scenarios) * 100
    opt_price = price_field[best_idx].item()
    opt_profit = profits[best_idx].item()

    # Динамический вывод в консоль
    click.secho(f"\n✅ Расчет проекта '{project_name}' завершен!", fg='green', bold=True)
    click.echo(f"Оптимальная цена: {opt_price:.2f} руб.")
    click.echo(f"Ожидаемая прибыль: {opt_profit:.2f} руб.")
    click.echo(f"Устойчивость модели: {success_rate:.2f}%")

    # --- Динамическая визуализация ---
    if plot:
        output_image = f"{project_name}_profit_landscape.png"
        plt.figure(figsize=(10, 6))
        indices = torch.randint(0, scenarios, (5000,))
        x = price_field[indices].cpu().numpy()
        y = profits[indices].cpu().numpy()
        
        plt.scatter(x, y, alpha=0.3, c=y, cmap='viridis')
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f"Ландшафт прибыли: {project_name.upper()} (2026)")
        plt.xlabel("Цена (руб)")
        plt.ylabel("Прибыль (руб)")
        
        plt.savefig(output_image)
        click.secho(f"График сохранен как: {output_image}", fg='yellow')

if __name__ == '__main__':
    run_holo_economy()
