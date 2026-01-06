import torch
import click
import yaml
import matplotlib.pyplot as plt

@click.command()
@click.option('--config', default='targetologist_config.yaml', help='Путь к YAML файлу')
@click.option('--plot', is_flag=True, help='Сгенерировать график')
def run_holo_economy_final(config, plot):
    with open(config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    m_p, p_p = params['market_params'], params['product_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenarios = m_p['scenarios_count']

    # --- Гибкая генерация поля конверсии (Сингум) ---
    if 'conversion_to_lead_min' in p_p:
        # Модель таргетолога: Лид -> Продажа
        conv_leads = p_p['conversion_to_lead_min'] + torch.rand(scenarios, device=device) * (p_p['conversion_to_lead_max'] - p_p['conversion_to_lead_min'])
        conv_sales = p_p['conversion_to_sale_min'] + torch.rand(scenarios, device=device) * (p_p['conversion_to_sale_max'] - p_p['conversion_to_sale_min'])
        conv_field = conv_leads * conv_sales
    else:
        # Базовая модель (психолог)
        conv_field = p_p['conversion_min'] + torch.rand(scenarios, device=device) * (p_p['conversion_max'] - p_p['conversion_min'])

    # --- Генерация остальных полей ---
    cpc_field = torch.normal(m_p['cpc_avg'], m_p['cpc_stdev'], (scenarios,), device=device)
    price_field = torch.linspace(p_p['price_range_min'], p_p['price_range_max'], scenarios, device=device)
    
    # LTV: если не задано (для таргетолога), считаем 1 продажу
    sess_min = p_p.get('repeat_sessions_min', 1)
    sess_max = p_p.get('repeat_sessions_max', 1)
    sessions_field = sess_min + torch.rand(scenarios, device=device) * (sess_max - sess_min)

    # --- Расчет модели ---
    def calculate(prices, cpc, conv, sess):
        new_clients = (m_p['budget'] / cpc) * conv
        total_revenue = new_clients * sess * prices
        unit_costs = (prices * p_p['tax_rate']) + p_p['base_cogs']
        total_costs = (new_clients * sess * unit_costs) + m_p['budget']
        profit = total_revenue - total_costs
        return profit

    with torch.inference_mode():
        profits = calculate(price_field, cpc_field, conv_field, sessions_field)

    # --- Вывод ---
    best_idx = torch.argmax(profits)
    success_rate = (torch.sum(profits > 0).item() / scenarios) * 100

    click.secho(f"\n✅ Расчет завершен успешно!", fg='green')
    click.echo(f"Оптимальная цена услуги: {price_field[best_idx]:.2f} руб.")
    click.echo(f"Макс. потенциальная прибыль: {profits[best_idx]:.2f} руб.")
    click.echo(f"Устойчивость модели: {success_rate:.2f}%")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(price_field.cpu().numpy()[::500], profits.cpu().numpy()[::500], alpha=0.1, c='blue')
        plt.title("Ландшафт прибыли (Holo-Economy 2026)")
        plt.xlabel("Цена услуги (руб)")
        plt.ylabel("Прибыль (руб)")
        plt.savefig('targetologist_profit.png')
        click.secho("График 'targetologist_profit.png' создан.", fg='yellow')

if __name__ == '__main__':
    run_holo_economy_final()
