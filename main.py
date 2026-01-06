# main_enhanced.py
import torch
import click
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
import pandas as pd

@click.command()
@click.option('--config', required=True, help='Путь к YAML файлу (например, psychologist_config.yaml)')
@click.option('--plot', is_flag=True, help='Сгенерировать график ландшафта прибыли')
@click.option('--report', is_flag=True, help='Создать детальный отчет в CSV')
def run_holo_economy(config, plot, report):
    # Извлекаем имя конфига для именования файлов
    config_path = Path(config)
    project_name = config_path.stem 

    with open(config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    m_p, p_p = params['market_params'], params['product_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenarios = int(m_p['scenarios_count'])

    # --- Генерация случайных полей ---
    if 'conversion_to_lead_min' in p_p:
        c_l = torch.distributions.Uniform(p_p['conversion_to_lead_min'], p_p['conversion_to_lead_max']).sample((scenarios,)).to(device)
        c_s = torch.distributions.Uniform(p_p['conversion_to_sale_min'], p_p['conversion_to_sale_max']).sample((scenarios,)).to(device)
        conv_field = c_l * c_s
    else:
        conv_field = torch.distributions.Uniform(p_p['conversion_min'], p_p['conversion_max']).sample((scenarios,)).to(device)

    cpc_field = torch.normal(m_p['cpc_avg'], m_p['cpc_stdev'], size=(scenarios,), device=device).clamp(min=1.0)
    price_field = torch.distributions.Uniform(p_p['price_range_min'], p_p['price_range_max']).sample((scenarios,)).to(device)
    
    # --- Генерация повторных сессий (LTV) ---
    if 'repeat_sessions_min' in p_p and 'repeat_sessions_max' in p_p:
        # Бета-распределение для более реалистичного LTV (большинство клиентов на 2-5 сессиях)
        alpha, beta = 2.0, 5.0  # Параметры для смещенного распределения
        repeat_sessions = torch.distributions.Beta(alpha, beta).sample((scenarios,)).to(device)
        # Масштабируем к диапазону repeat_sessions_min - repeat_sessions_max
        repeat_sessions = p_p['repeat_sessions_min'] + repeat_sessions * (p_p['repeat_sessions_max'] - p_p['repeat_sessions_min'])
    else:
        repeat_sessions = torch.ones((scenarios,), device=device)

    # --- Расчет с учетом LTV ---
    with torch.inference_mode():
        # Базовые клиенты (первые сессии)
        initial_clients = (m_p['budget'] / cpc_field) * conv_field
        
        # Общее количество сессий с учетом повторных визитов
        total_sessions = initial_clients * repeat_sessions
        
        # Маржа с одной сессии
        unit_margin = price_field - (price_field * p_p['tax_rate']) - p_p['base_cogs']
        
        # Прибыль с учетом всех сессий
        revenue = total_sessions * unit_margin
        profits = revenue - m_p['budget']
        
        # Дополнительная аналитика
        cac = m_p['budget'] / initial_clients  # Customer Acquisition Cost
        ltv = repeat_sessions * unit_margin     # Lifetime Value на клиента
        ltv_cac_ratio = ltv / cac               # Отношение LTV к CAC

    # --- Расширенная аналитика ---
    best_idx = torch.argmax(profits)
    worst_idx = torch.argmin(profits)
    success_rate = (torch.sum(profits > 0).item() / scenarios) * 100
    median_profit = torch.median(profits).item()
    profit_std = torch.std(profits).item()
    
    # Процентили прибыли
    profits_np = profits.cpu().numpy()
    profit_percentiles = {
        '5%': np.percentile(profits_np, 5),
        '25%': np.percentile(profits_np, 25),
        '50%': np.percentile(profits_np, 50),
        '75%': np.percentile(profits_np, 75),
        '95%': np.percentile(profits_np, 95)
    }
    
    # Оптимальные параметры
    opt_price = price_field[best_idx].item()
    opt_profit = profits[best_idx].item()
    opt_conversion = conv_field[best_idx].item() * 100
    opt_cpc = cpc_field[best_idx].item()
    opt_repeat_sessions = repeat_sessions[best_idx].item()
    opt_ltv_cac = ltv_cac_ratio[best_idx].item()
    
    # Средние значения по прибыльным сценариям
    profitable_mask = profits > 0
    if torch.any(profitable_mask):
        avg_profitable_price = price_field[profitable_mask].mean().item()
        avg_profitable_repeat = repeat_sessions[profitable_mask].mean().item()
        avg_profitable_ltv_cac = ltv_cac_ratio[profitable_mask].mean().item()
    else:
        avg_profitable_price = avg_profitable_repeat = avg_profitable_ltv_cac = 0

    # --- Детальный вывод в консоль ---
    click.secho(f"\n📊 РАСЧЕТ ПРОЕКТА: '{project_name.upper()}'", fg='cyan', bold=True)
    click.secho("=" * 60, fg='cyan')
    
    click.secho("\n🎯 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:", fg='green', bold=True)
    click.echo(f"  Цена за сессию: {opt_price:,.2f} руб.")
    click.echo(f"  Ожидаемая прибыль: {opt_profit:,.0f} руб.")
    click.echo(f"  Конверсия: {opt_conversion:.2f}%")
    click.echo(f"  CPC: {opt_cpc:.1f} руб.")
    click.echo(f"  Повторных сессий на клиента: {opt_repeat_sessions:.1f}")
    click.echo(f"  LTV/CAC: {opt_ltv_cac:.2f}")
    
    click.secho("\n📈 СТАТИСТИКА ПРИБЫЛИ:", fg='yellow', bold=True)
    click.echo(f"  Успешных сценариев: {success_rate:.2f}%")
    click.echo(f"  Медианная прибыль: {median_profit:,.0f} руб.")
    click.echo(f"  Стандартное отклонение: {profit_std:,.0f} руб.")
    
    click.secho("\n📊 ПРОЦЕНТИЛИ ПРИБЫЛИ:", fg='blue', bold=True)
    for perc, value in profit_percentiles.items():
        click.echo(f"  {perc}: {value:,.0f} руб.")
    
    click.secho("\n💡 РЕКОМЕНДАЦИИ:", fg='magenta', bold=True)
    if success_rate < 30:
        click.secho("  ⚠️  ВЫСОКИЙ РИСК: Менее 30% успешных сценариев", fg='red')
        click.echo("  • Увеличьте бюджет на тестирование")
        click.echo("  • Снизьте CPC через улучшение качества трафика")
        click.echo("  • Повышайте конверсию через оптимизацию лендинга")
    elif success_rate < 70:
        click.secho("  ⚠️  СРЕДНИЙ РИСК: 30-70% успешных сценариев", fg='yellow')
        click.echo("  • Сфокусируйтесь на улучшении LTV через удержание клиентов")
        click.echo("  • Тестируйте повышение цены постепенно")
    else:
        click.secho("  ✅ НИЗКИЙ РИСК: Более 70% успешных сценариев", fg='green')
        click.echo("  • Масштабируйте рекламные кампании")
        click.echo("  • Рассмотрите увеличение цены")
    
    if avg_profitable_ltv_cac > 0:
        if avg_profitable_ltv_cac < 3:
            click.secho(f"  ⚠️  LTV/CAC {avg_profitable_ltv_cac:.1f} ниже нормы 3.0", fg='yellow')
        else:
            click.secho(f"  ✅ LTV/CAC {avg_profitable_ltv_cac:.1f} на хорошем уровне", fg='green')

    # --- Детальная визуализация ---
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Ландшафт прибыли
        indices = torch.randint(0, scenarios, (min(10000, scenarios),))
        x = price_field[indices].cpu().numpy()
        y = profits[indices].cpu().numpy()
        
        scatter = axes[0, 0].scatter(x, y, alpha=0.4, c=y, cmap='RdYlGn', s=10)
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(opt_price, color='blue', linestyle='--', alpha=0.5, label=f'Опт. цена: {opt_price:.0f} руб.')
        axes[0, 0].set_xlabel("Цена за сессию (руб)")
        axes[0, 0].set_ylabel("Прибыль (руб)")
        axes[0, 0].set_title(f"Ландшафт прибыли")
        axes[0, 0].legend()
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. Распределение прибыли
        profitable_profits = profits[profits > 0].cpu().numpy()
        if len(profitable_profits) > 0:
            axes[0, 1].hist(profitable_profits, bins=50, color='green', alpha=0.7, label='Прибыльные')
        if len(profits_np[profits_np <= 0]) > 0:
            axes[0, 1].hist(profits_np[profits_np <= 0], bins=50, color='red', alpha=0.7, label='Убыточные')
        axes[0, 1].axvline(median_profit, color='black', linestyle='--', label=f'Медиана: {median_profit:,.0f}')
        axes[0, 1].set_xlabel("Прибыль (руб)")
        axes[0, 1].set_ylabel("Частота")
        axes[0, 1].set_title(f"Распределение прибыли (успешность: {success_rate:.1f}%)")
        axes[0, 1].legend()
        
        # 3. Зависимость прибыли от повторных сессий
        axes[1, 0].scatter(repeat_sessions[indices].cpu().numpy(), 
                          profits[indices].cpu().numpy(), 
                          alpha=0.3, s=10)
        axes[1, 0].set_xlabel("Повторных сессий на клиента")
        axes[1, 0].set_ylabel("Прибыль (руб)")
        axes[1, 0].set_title("Влияние LTV на прибыль")
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Box plot прибыли по ценовым квантилям
        price_quantiles = pd.qcut(price_field.cpu().numpy(), 5)
        profit_df = pd.DataFrame({
            'price_quantile': price_quantiles,
            'profit': profits_np
        })
        
        boxes = []
        labels = []
        for quantile in profit_df['price_quantile'].cat.categories:
            box_data = profit_df[profit_df['price_quantile'] == quantile]['profit'].values
            boxes.append(box_data)
            labels.append(str(quantile))
        
        bp = axes[1, 1].boxplot(boxes, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'gold', 'orange', 'salmon']):
            patch.set_facecolor(color)
        axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel("Ценовые квинтили")
        axes[1, 1].set_ylabel("Прибыль (руб)")
        axes[1, 1].set_title("Распределение прибыли по ценовым группам")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f"Детальный анализ: {project_name.upper()} (2026)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_image = f"{project_name}_detailed_analysis.png"
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        click.secho(f"\n📊 График сохранен как: {output_image}", fg='yellow')

    # --- Детальный отчет в CSV ---
    if report:
        # Выборка для отчета (первые 10,000 сценариев)
        sample_size = min(10000, scenarios)
        indices = torch.randperm(scenarios)[:sample_size]
        
        report_data = {
            'цена': price_field[indices].cpu().numpy(),
            'прибыль': profits[indices].cpu().numpy(),
            'конверсия_%': conv_field[indices].cpu().numpy() * 100,
            'cpc': cpc_field[indices].cpu().numpy(),
            'повторные_сессии': repeat_sessions[indices].cpu().numpy(),
            'ltv_cac': ltv_cac_ratio[indices].cpu().numpy(),
            'клиенты': initial_clients[indices].cpu().numpy(),
            'статус': np.where(profits[indices].cpu().numpy() > 0, 'прибыль', 'убыток')
        }
        
        df = pd.DataFrame(report_data)
        csv_file = f"{project_name}_detailed_report.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # Сводная статистика
        summary_stats = {
            'параметр': ['оптимальная_цена', 'макс_прибыль', 'успешность_%', 'медиана_прибыли', 
                        'средний_ltv_cac', 'средние_повторные_сессии'],
            'значение': [opt_price, opt_profit, success_rate, median_profit, 
                        avg_profitable_ltv_cac, avg_profitable_repeat]
        }
        df_summary = pd.DataFrame(summary_stats)
        summary_file = f"{project_name}_summary_stats.csv"
        df_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        click.secho(f"📄 Детальный отчет сохранен как: {csv_file}", fg='green')
        click.secho(f"📊 Сводная статистика: {summary_file}", fg='green')

if __name__ == '__main__':
    run_holo_economy()