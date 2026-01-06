# main_fixed.py
import torch
import click
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

@click.command()
@click.option('--config', required=True, help='Путь к YAML файлу')
@click.option('--plot', is_flag=True, help='Сгенерировать график')
@click.option('--robust', is_flag=True, help='Использовать устойчивые оценки вместо выбросов')
def run_holo_economy(config, plot, robust):
    config_path = Path(config)
    project_name = config_path.stem 

    with open(config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    m_p, p_p = params['market_params'], params['product_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenarios = min(int(m_p['scenarios_count']), 1000000)  # Ограничим для скорости

    # --- ИСПРАВЛЕННАЯ ГЕНЕРАЦИЯ ---
    
    # 1. CPC: используем гамма-распределение вместо нормального (неотрицательное, реалистичное)
    # Параметры гамма-распределения: shape=k, scale=theta, mean=k*theta, var=k*theta^2
    cpc_mean = m_p['cpc_avg']
    cpc_std = m_p['cpc_stdev']
    # Выбираем параметры чтобы избежать значений < 50 руб
    cpc_shape = (cpc_mean / cpc_std) ** 2  # k
    cpc_scale = (cpc_std ** 2) / cpc_mean  # theta
    cpc_field = torch.distributions.Gamma(cpc_shape, 1/cpc_scale).sample((scenarios,)).to(device)
    cpc_field = torch.clamp(cpc_field, min=100.0, max=400.0)  # Реалистичный диапазон
    
    # 2. Конверсия: логнормальное распределение (больше значений внизу)
    conv_mu = torch.log(torch.tensor(0.015))  # медиана ~1.5%
    conv_sigma = 0.5
    conv_field = torch.distributions.LogNormal(conv_mu, conv_sigma).sample((scenarios,)).to(device)
    conv_field = torch.clamp(conv_field, min=0.005, max=0.035)  # 0.5%-3.5%
    
    # 3. Цена: нормальное распределение с центром в середине диапазона
    price_center = (p_p['price_range_min'] + p_p['price_range_max']) / 2
    price_std = (p_p['price_range_max'] - p_p['price_range_min']) / 6  # 99.7% в диапазоне
    price_field = torch.normal(price_center, price_std, size=(scenarios,), device=device)
    price_field = torch.clamp(price_field, min=p_p['price_range_min'], max=p_p['price_range_max'])
    
    # 4. Повторные сессии: усеченное нормальное распределение (большинство 2-5 сессий)
    repeat_center = 4.0
    repeat_std = 2.0
    repeat_sessions = torch.normal(repeat_center, repeat_std, size=(scenarios,), device=device)
    repeat_sessions = torch.clamp(repeat_sessions, min=2.0, max=15.0)
    repeat_sessions = torch.round(repeat_sessions)  # Целое число сессий

    # --- РАСЧЕТ С УЧЕТОМ LTV ---
    with torch.inference_mode():
        # Исправленный расчет с защитой от деления на 0
        cpc_safe = torch.clamp(cpc_field, min=1.0)
        initial_clients = (m_p['budget'] / cpc_safe) * conv_field
        initial_clients = torch.clamp(initial_clients, min=1.0)  # Минимум 1 клиент
        
        total_sessions = initial_clients * repeat_sessions
        unit_margin = price_field * (1 - p_p['tax_rate']) - p_p['base_cogs']
        
        # Прибыль от всех сессий за вычетом маркетингового бюджета
        revenue = total_sessions * unit_margin
        profits = revenue - m_p['budget']
        
        # Дополнительная аналитика
        cac = m_p['budget'] / initial_clients
        ltv = repeat_sessions * unit_margin
        ltv_cac_ratio = ltv / torch.clamp(cac, min=1.0)

    # --- УСТОЙЧИВАЯ АНАЛИТИКА ---
    
    # Фильтруем выбросы (удаляем топ и низ 1%)
    if robust:
        q_low = torch.quantile(profits, 0.01)
        q_high = torch.quantile(profits, 0.99)
        valid_mask = (profits >= q_low) & (profits <= q_high)
    else:
        valid_mask = torch.ones_like(profits, dtype=torch.bool)
    
    valid_profits = profits[valid_mask]
    valid_count = valid_mask.sum().item()
    
    if valid_count == 0:
        click.secho("❌ Нет валидных сценариев для анализа", fg='red')
        return
    
    # Находим параметры по медианному сценарию, а не максимальному
    median_idx = torch.argsort(valid_profits)[valid_count // 2]
    median_profit = valid_profits[median_idx].item()
    
    # Ищем лучший сценарий в пределах 90-го процентиля (без крайних выбросов)
    profit_90 = torch.quantile(valid_profits, 0.90)
    best_in_range_mask = (profits <= profit_90) & valid_mask
    if best_in_range_mask.any():
        best_idx = torch.argmax(profits[best_in_range_mask])
        best_mask_indices = torch.where(best_in_range_mask)[0]
        best_global_idx = best_mask_indices[best_idx]
    else:
        best_global_idx = torch.argmax(valid_profits)
    
    success_rate = (torch.sum(valid_profits > 0).item() / valid_count) * 100
    
    # Оптимальные параметры (по лучшему в разумном диапазоне)
    opt_price = price_field[best_global_idx].item()
    opt_profit = profits[best_global_idx].item()
    opt_conversion = conv_field[best_global_idx].item() * 100
    opt_cpc = cpc_field[best_global_idx].item()
    opt_repeat_sessions = repeat_sessions[best_global_idx].item()
    opt_ltv_cac = ltv_cac_ratio[best_global_idx].item()
    
    # Медианные значения по прибыльным сценариям
    profitable_mask = valid_profits > 0
    if profitable_mask.any():
        profitable_indices = torch.where(valid_mask)[0][profitable_mask]
        median_profitable_price = price_field[profitable_indices].median().item()
        median_profitable_repeat = repeat_sessions[profitable_indices].median().item()
        median_profitable_ltv_cac = ltv_cac_ratio[profitable_indices].median().item()
        median_profitable_profit = valid_profits[profitable_mask].median().item()
    else:
        median_profitable_price = median_profitable_repeat = median_profitable_ltv_cac = median_profitable_profit = 0
    
    # Статистика
    median_profit_all = torch.median(valid_profits).item()
    profit_std = torch.std(valid_profits).item()
    
    profits_np = valid_profits.cpu().numpy()
    profit_percentiles = {
        '5%': np.percentile(profits_np, 5),
        '25%': np.percentile(profits_np, 25),
        '50%': np.percentile(profits_np, 50),
        '75%': np.percentile(profits_np, 75),
        '95%': np.percentile(profits_np, 95)
    }
    
    # --- ИСПРАВЛЕННЫЙ ВЫВОД ---
    click.secho(f"\n📊 РЕАЛИСТИЧНЫЙ РАСЧЕТ: '{project_name.upper()}'", fg='cyan', bold=True)
    click.secho("=" * 60, fg='cyan')
    
    click.secho(f"\n📈 СТАТИСТИКА ({valid_count:,} сценариев):", fg='yellow', bold=True)
    click.echo(f"  Успешных сценариев: {success_rate:.1f}%")
    click.echo(f"  Медианная прибыль: {median_profit_all:,.0f} руб.")
    click.echo(f"  Стандартное отклонение: {profit_std:,.0f} руб.")
    
    click.secho("\n🎯 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ (без экстремальных выбросов):", fg='green', bold=True)
    click.echo(f"  Цена за сессию: {opt_price:,.0f} руб.")
    click.echo(f"  Ожидаемая прибыль: {opt_profit:,.0f} руб.")
    click.echo(f"  Конверсия: {opt_conversion:.2f}%")
    click.echo(f"  CPC: {opt_cpc:.0f} руб.")
    click.echo(f"  Повторных сессий: {opt_repeat_sessions:.0f}")
    click.echo(f"  LTV/CAC: {opt_ltv_cac:.1f}")
    
    click.secho("\n📊 СРЕДНИЕ ПО ПРИБЫЛЬНЫМ СЦЕНАРИЯМ:", fg='blue', bold=True)
    if profitable_mask.any():
        click.echo(f"  Цена: {median_profitable_price:,.0f} руб.")
        click.echo(f"  Повторные сессии: {median_profitable_repeat:.1f}")
        click.echo(f"  LTV/CAC: {median_profitable_ltv_cac:.1f}")
        click.echo(f"  Прибыль: {median_profitable_profit:,.0f} руб.")
    else:
        click.secho("  Нет прибыльных сценариев", fg='red')
    
    click.secho("\n📊 ПРОЦЕНТИЛИ ПРИБЫЛИ:", fg='magenta', bold=True)
    for perc, value in profit_percentiles.items():
        color = 'green' if value > 0 else 'red' if value < -10000 else 'yellow'
        click.secho(f"  {perc}: {value:,.0f} руб.", fg=color)
    
    click.secho("\n💡 ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:", fg='cyan', bold=True)
    if success_rate < 20:
        click.secho("  🔴 КРИТИЧЕСКИЙ РИСК", fg='red')
        click.echo("  • Увеличьте бюджет минимум до 100,000 руб.")
        click.echo("  • Снижайте CPC через таргетинг и ретаргетинг")
        click.echo("  • Повышайте конверсию лендинга до 2%+")
    elif success_rate < 50:
        click.secho("  🟡 СРЕДНИЙ РИСК", fg='yellow')
        click.echo("  • Оптимальная цена: 3,500-4,500 руб.")
        click.echo("  • Цель: 4+ сессии на клиента через систему напоминаний")
        click.echo("  • Тестируйте нишевые специализации")
    else:
        click.secho("  🟢 НИЗКИЙ РИСК", fg='green')
        click.echo("  • Можно тестировать масштабирование")
        click.echo("  • Рассмотрите премиум-сегмент (5,000+ руб.)")
        click.echo("  • Инвестируйте в супервизии и обучение")
    
    if profitable_mask.any() and median_profitable_ltv_cac < 3:
        click.secho(f"  ⚠️  LTV/CAC {median_profitable_ltv_cac:.1f} ниже нормы 3.0", fg='yellow')
        click.echo("  • Увеличивайте повторные продажи")
        click.echo("  • Внедряйте программы лояльности")
        click.echo("  • Повышайте качество сервиса")

    # --- ИСПРАВЛЕННАЯ ВИЗУАЛИЗАЦИЯ ---
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Отбираем 5000 случайных сценариев для визуализации
        sample_size = min(5000, valid_count)
        sample_indices = torch.randperm(valid_count)[:sample_size]
        sample_profits = valid_profits[sample_indices].cpu().numpy()
        sample_prices = price_field[valid_mask][sample_indices].cpu().numpy()
        sample_repeats = repeat_sessions[valid_mask][sample_indices].cpu().numpy()
        
        # 1. Прибыль vs Цена
        scatter1 = axes[0, 0].scatter(sample_prices, sample_profits, 
                                     c=sample_profits, cmap='RdYlGn', 
                                     alpha=0.6, s=20, vmin=-50000, vmax=50000)
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].axvline(opt_price, color='blue', linestyle='--', alpha=0.7, 
                          label=f'Опт. цена: {opt_price:,.0f} руб.')
        axes[0, 0].set_xlabel("Цена за сессию (руб)")
        axes[0, 0].set_ylabel("Прибыль (руб)")
        axes[0, 0].set_title(f"Прибыль vs Цена (успешность: {success_rate:.1f}%)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Распределение прибыли
        profit_min = max(sample_profits.min(), -50000)
        profit_max = min(sample_profits.max(), 50000)
        bins = np.linspace(profit_min, profit_max, 50)
        
        axes[0, 1].hist(sample_profits[sample_profits > 0], bins=bins[bins > 0], 
                       color='green', alpha=0.7, label='Прибыльные')
        axes[0, 1].hist(sample_profits[sample_profits <= 0], bins=bins[bins <= 0], 
                       color='red', alpha=0.7, label='Убыточные')
        axes[0, 1].axvline(median_profit_all, color='black', linestyle='--', 
                          label=f'Медиана: {median_profit_all:,.0f} руб.')
        axes[0, 1].axvline(0, color='red', linestyle='-', alpha=0.5)
        axes[0, 1].set_xlabel("Прибыль (руб)")
        axes[0, 1].set_ylabel("Количество сценариев")
        axes[0, 1].set_title("Распределение прибыли")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Прибыль vs Повторные сессии
        axes[1, 0].scatter(sample_repeats, sample_profits, 
                          c=sample_profits, cmap='RdYlGn', 
                          alpha=0.6, s=20, vmin=-50000, vmax=50000)
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel("Повторных сессий на клиента")
        axes[1, 0].set_ylabel("Прибыль (руб)")
        axes[1, 0].set_title("Влияние LTV на прибыль")
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Распределение ключевых параметров
        params_to_plot = ['CPC', 'Конверсия', 'Повторные сессии']
        param_values = [
            cpc_field[valid_mask][sample_indices].cpu().numpy(),
            conv_field[valid_mask][sample_indices].cpu().numpy() * 100,
            repeat_sessions[valid_mask][sample_indices].cpu().numpy()
        ]
        
        boxes = []
        labels = params_to_plot
        for vals in param_values:
            boxes.append(vals)
        
        bp = axes[1, 1].boxplot(boxes, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'gold']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[1, 1].set_ylabel("Значение")
        axes[1, 1].set_title("Распределение ключевых параметров")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Реалистичный анализ: {project_name.upper()} (2026)", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_image = f"{project_name}_realistic_analysis.png"
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        click.secho(f"\n📊 График сохранен как: {output_image}", fg='yellow')
        plt.close()

if __name__ == '__main__':
    run_holo_economy()