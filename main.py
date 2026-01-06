import torch
import click
import time
import yaml
import matplotlib.pyplot as plt

@click.command()
@click.option('--config', default='psychologist_config.yaml', help='Путь к YAML файлу')
@click.option('--plot', is_flag=True, help='Сгенерировать график ландшафта прибыли')
def run_holo_economy_final(config, plot):
    # Явно указываем utf-8, чтобы читать кириллицу в комментариях YAML
    with open(config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    m_p, p_p = params['market_params'], params['product_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenarios = m_p['scenarios_count']

    # --- Генерация полей (Сингумов) ---
    conv_field = p_p['conversion_min'] + torch.rand(scenarios, device=device) * (p_p['conversion_max'] - p_p['conversion_min'])
    cpc_field = torch.normal(m_p['cpc_avg'], m_p['cpc_stdev'], (scenarios,), device=device)
    price_field = torch.linspace(p_p['price_range_min'], p_p['price_range_max'], scenarios, device=device)
    
    # Поле сессий (LTV) для психолога
    sessions_field = p_p['repeat_sessions_min'] + torch.rand(scenarios, device=device) * (p_p['repeat_sessions_max'] - p_p['repeat_sessions_min'])

    # --- Расчет модели ---
    def calculate(prices, cpc, conv, sess):
        new_clients = (m_p['budget'] / cpc) * conv
        total_revenue = new_clients * sess * prices
        
        # Расходы на единицу (налог + аренда/сервис)
        unit_costs = (prices * p_p['tax_rate']) + p_p['base_cogs']
        total_costs = (new_clients * sess * unit_costs) + m_p['budget']
        
        profit = total_revenue - total_costs
        margin = profit / (new_clients + 1e-6)
        return profit, margin

    with torch.inference_mode():
        profits, margins = calculate(price_field, cpc_field, conv_field, sessions_field)

    # --- Аналитика ---
    best_idx = torch.argmax(profits)
    success_rate = (torch.sum(profits > 0).item() / scenarios) * 100

    click.secho(f"\n✅ Расчет завершен успешно!", fg='green')
    click.echo(f"Оптимальная цена сессии: {price_field[best_idx]:.2f} руб.")
    click.echo(f"Макс. потенциальная прибыль: {profits[best_idx]:.2f} руб.")
    click.echo(f"Вероятность окупаемости (устойчивость): {success_rate:.2f}%")

    if plot:
        x = price_field.cpu().numpy()
        y = profits.cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.scatter(x[::100], y[::100], c=y[::100], cmap='plasma', alpha=0.05)
        plt.title("Ландшафт прибыли психолога (Holo-Economy)")
        plt.xlabel("Цена за сессию (руб)")
        plt.ylabel("Прибыль (руб)")
        plt.grid(True)
        plt.savefig('psychologist_profit.png')
        click.secho("График 'psychologist_profit.png' успешно создан.", fg='yellow')

if __name__ == '__main__':
    run_holo_economy_final()
