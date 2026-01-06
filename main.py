# main_fixed.py
import torch
import click
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--config', required=True, help='Путь к YAML файлу')
@click.option('--plot', is_flag=True, help='Сгенерировать график')
def run_psychologist_economy(config, plot):
    config_path = Path(config)
    project_name = config_path.stem 

    with open(config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    m_p, p_p = params['market_params'], params['product_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenarios = min(int(m_p['scenarios_count']), 500000)

    # --- РЕАЛИСТИЧНАЯ ГЕНЕРАЦИЯ СЦЕНАРИЕВ ---
    
    # 1. CPC: логнормальное распределение (не бывает отрицательных CPC)
    cpc_log_mean = torch.log(torch.tensor(m_p['cpc_avg']))
    cpc_log_std = 0.3  # умеренная волатильность
    cpc_field = torch.distributions.LogNormal(cpc_log_mean, cpc_log_std).sample((scenarios,)).to(device)
    cpc_field = torch.clamp(cpc_field, min=100.0, max=350.0)
    
    # 2. Конверсия: бета-распределение (большинство значений в середине)
    # Параметры для пика на ~1.5%
    alpha, beta = 2.0, 8.0
    conv_beta = torch.distributions.Beta(alpha, beta).sample((scenarios,)).to(device)
    # Масштабируем к нужному диапазону
    conv_min, conv_max = p_p['conversion_min'], p_p['conversion_max']
    conv_field = conv_min + conv_beta * (conv_max - conv_min)
    
    # 3. Цена: нормальное распределение
    price_center = (p_p['price_range_min'] + p_p['price_range_max']) / 2
    price_std = (p_p['price_range_max'] - p_p['price_range_min']) / 4
    price_field = torch.normal(price_center, price_std, size=(scenarios,), device=device)
    price_field = torch.clamp(price_field, min=p_p['price_range_min'], max=p_p['price_range_max'])
    
    # 4. Повторные сессии: усеченное нормальное распределение
    repeat_mean = 3.5  # Среднее 3.5 сессии на клиента
    repeat_std = 2.0
    repeat_sessions = torch.normal(repeat_mean, repeat_std, size=(scenarios,), device=device)
    repeat_sessions = torch.clamp(repeat_sessions, min=1.0, max=12.0)
    repeat_sessions = torch.round(repeat_sessions)  # Целое число сессий

    # 5. Эффективный бюджет (не весь тратится эффективно)
    effective_budget = m_p['budget'] * 0.8

    # --- РАСЧЕТ ЭКОНОМИКИ ---
    with torch.inference_mode():
        # Клики
        clicks = effective_budget / cpc_field
        
        # Клиенты (первые сессии)
        initial_clients = clicks * conv_field
        initial_clients = torch.clamp(initial_clients, min=0.1)  # Не менее 0.1 клиента
        
        # Общее количество сессий с учетом повторных
        total_sessions = initial_clients * repeat_sessions
        
        # Маржа за сессию
        unit_margin = price_field * (1 - p_p['tax_rate']) - p_p['base_cogs']
        
        # Прибыль
        revenue = total_sessions * unit_margin
        profits = revenue - effective_budget
        
        # CAC и LTV
        cac = effective_budget / initial_clients
        ltv = repeat_sessions * unit_margin
        ltv_cac_ratio = ltv / torch.clamp(cac, min=1.0)

    # --- ФИЛЬТРАЦИЯ РЕАЛИСТИЧНЫХ СЦЕНАРИЕВ ---
    # Убираем явно нереалистичные сценарии
    realistic_mask = (
        (conv_field >= 0.005) & 
        (conv_field <= 0.05) & 
        (cpc_field >= 80) & 
        (cpc_field <= 400) &
        (initial_clients >= 0.5) &  # Хотя бы полклиента
        (torch.isfinite(profits))   # Убираем NaN/Inf
    )
    
    if not realistic_mask.any():
        click.secho("❌ Нет реалистичных сценариев! Проверьте параметры конфига.", fg='red')
        return
    
    realistic_indices = torch.where(realistic_mask)[0]
    realistic_profits = profits[realistic_indices]
    realistic_count = len(realistic_indices)
    
    # --- АНАЛИТИКА ---
    
    # Базовая статистика
    success_rate = (realistic_profits > 0).sum().item() / realistic_count * 100
    median_profit = torch.median(realistic_profits).item()
    mean_profit = torch.mean(realistic_profits).item()
    
    # Находим оптимальный сценарий (без экстремальных выбросов)
    # Берем 90-й процентиль прибыли как верхнюю границу
    profit_90 = torch.quantile(realistic_profits, 0.90)
    good_profits_mask = realistic_profits <= profit_90
    
    if good_profits_mask.any():
        # Находим лучший среди хороших сценариев
        best_good_idx = torch.argmax(realistic_profits[good_profits_mask])
        # Получаем глобальный индекс
        good_indices = realistic_indices[good_profits_mask]
        best_idx = good_indices[best_good_idx]
    else:
        best_idx = torch.argmax(profits)
    
    # Параметры оптимального сценария
    opt_price = price_field[best_idx].item()
    opt_profit = profits[best_idx].item()
    opt_conversion = conv_field[best_idx].item() * 100
    opt_cpc = cpc_field[best_idx].item()
    opt_repeat = repeat_sessions[best_idx].item()
    opt_ltv_cac = ltv_cac_ratio[best_idx].item()
    
    # Анализ прибыльных сценариев
    profitable_mask_realistic = realistic_profits > 0
    if profitable_mask_realistic.any():
        profitable_indices = realistic_indices[profitable_mask_realistic]
        median_profitable_price = price_field[profitable_indices].median().item()
        median_profitable_repeat = repeat_sessions[profitable_indices].median().item()
        median_profitable_ltv_cac = ltv_cac_ratio[profitable_indices].median().item()
        median_profitable_profit = realistic_profits[profitable_mask_realistic].median().item()
    else:
        median_profitable_price = median_profitable_repeat = median_profitable_ltv_cac = median_profitable_profit = 0
    
    # Процентили прибыли
    profits_np = realistic_profits.cpu().numpy()
    profit_percentiles = {
        '5%': np.percentile(profits_np, 5),
        '25%': np.percentile(profits_np, 25),
        '50%': np.percentile(profits_np, 50),
        '75%': np.percentile(profits_np, 75),
        '95%': np.percentile(profits_np, 95)
    }
    
    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    click.secho(f"\n🧠 АНАЛИЗ", fg='cyan', bold=True)
    click.secho("=" * 60, fg='cyan')
    
    click.secho(f"\n📊 ОБЩАЯ СТАТИСТИКА ({realistic_count:,} сценариев):", fg='yellow', bold=True)
    click.echo(f"  Успешных сценариев: {success_rate:.1f}%")
    click.echo(f"  Средняя прибыль: {mean_profit:,.0f} руб.")
    click.echo(f"  Медианная прибыль: {median_profit:,.0f} руб.")
    
    click.secho("\n🎯 РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ:", fg='green', bold=True)
    click.echo(f"  Цена за сессию: {opt_price:,.0f} руб.")
    click.echo(f"  Ожидаемая прибыль: {opt_profit:,.0f} руб.")
    click.echo(f"  Конверсия: {opt_conversion:.2f}%")
    click.echo(f"  CPC: {opt_cpc:.0f} руб.")
    click.echo(f"  Среднее количество сессий: {opt_repeat:.1f}")
    click.echo(f"  LTV/CAC: {opt_ltv_cac:.1f}")
    
    if profitable_mask_realistic.any():
        click.secho("\n📊 ХАРАКТЕРИСТИКИ ПРИБЫЛЬНЫХ СЦЕНАРИЕВ:", fg='blue', bold=True)
        click.echo(f"  Типичная цена: {median_profitable_price:,.0f} руб.")
        click.echo(f"  Типичное количество сессий: {median_profitable_repeat:.1f}")
        click.echo(f"  Типичный LTV/CAC: {median_profitable_ltv_cac:.1f}")
        click.echo(f"  Типичная прибыль: {median_profitable_profit:,.0f} руб.")
    
    click.secho("\n📈 ВЕРОЯТНОСТНЫЕ ДИАПАЗОНЫ:", fg='magenta', bold=True)
    for perc, value in profit_percentiles.items():
        if value > 0:
            click.secho(f"  {perc}: прибыль {value:,.0f} руб.", fg='green')
        elif value < -20000:
            click.secho(f"  {perc}: убыток {abs(value):,.0f} руб.", fg='red')
        else:
            click.secho(f"  {perc}: {value:,.0f} руб.", fg='yellow')
    
    click.secho("\n💡 РЕКОМЕНДАЦИИ:", fg='cyan', bold=True)
    
    if success_rate < 25:
        click.secho("  🔴 ВЫСОКИЙ РИСК", fg='red')
        click.echo("    1. Увеличьте бюджет до 100,000+ руб.")
        click.echo("    2. Снижайте CPC через SEO и прямой трафик")
        click.echo("    3. Работайте над конверсией (отзывы, доверие)")
    elif success_rate < 50:
        click.secho("  🟡 СРЕДНИЙ РИСК", fg='yellow')
        click.echo(f"    1. Оптимальная цена: 3,500-4,500 руб.")
        click.echo("    2. Цель: 4+ сессии на клиента")
        click.echo("    3. Предлагайте пакеты сессий")
    else:
        click.secho("  🟢 НИЗКИЙ РИСК", fg='green')
        click.echo("    1. Можно тестировать повышение цены")
        click.echo("    2. Масштабируйте успешные каналы")
        click.echo("    3. Инвестируйте в обучение")
    
    if median_profitable_ltv_cac > 0 and median_profitable_ltv_cac < 3:
        click.secho(f"\n⚠️  LTV/CAC {median_profitable_ltv_cac:.1f} ниже нормы 3.0", fg='yellow')
        click.echo("    • Увеличивайте количество сессий на клиента")
        click.echo("    • Внедряйте систему напоминаний")
        click.echo("    • Улучшайте качество сервиса")

    # --- ВИЗУАЛИЗАЦИЯ ---
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Выборка для визуализации
        sample_size = min(3000, realistic_count)
        sample_indices = np.random.choice(realistic_count, sample_size, replace=False)
        
        sample_profits = realistic_profits[sample_indices].cpu().numpy()
        sample_prices = price_field[realistic_indices][sample_indices].cpu().numpy()
        sample_repeats = repeat_sessions[realistic_indices][sample_indices].cpu().numpy()
        sample_cpc = cpc_field[realistic_indices][sample_indices].cpu().numpy()
        sample_conv = conv_field[realistic_indices][sample_indices].cpu().numpy() * 100
        
        # 1. Прибыль vs Цена
        colors = ['red' if p <= 0 else 'green' for p in sample_profits]
        axes[0, 0].scatter(sample_prices, sample_profits, c=colors, alpha=0.5, s=20)
        axes[0, 0].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].axvline(opt_price, color='blue', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel("Цена за сессию, руб")
        axes[0, 0].set_ylabel("Прибыль, руб")
        axes[0, 0].set_title(f"Прибыль vs Цена (успешность: {success_rate:.1f}%)")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Распределение прибыли
        axes[0, 1].hist(sample_profits, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(0, color='red', linestyle='-', linewidth=2)
        axes[0, 1].axvline(median_profit, color='green', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel("Прибыль, руб")
        axes[0, 1].set_ylabel("Частота")
        axes[0, 1].set_title("Распределение прибыли")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Влияние повторных сессий
        unique_repeats = np.unique(sample_repeats.astype(int))
        repeat_groups = []
        repeat_labels = []
        
        for rep in unique_repeats:
            mask = sample_repeats == rep
            if np.sum(mask) > 5:
                repeat_groups.append(sample_profits[mask])
                repeat_labels.append(str(int(rep)))
        
        if repeat_groups:
            box_plot = axes[1, 0].boxplot(repeat_groups, labels=repeat_labels, patch_artist=True)
            for box in box_plot['boxes']:
                box.set_facecolor('lightblue')
            axes[1, 0].axhline(0, color='red', linestyle='-', alpha=0.5)
            axes[1, 0].set_xlabel("Количество сессий на клиента")
            axes[1, 0].set_ylabel("Прибыль, руб")
            axes[1, 0].set_title("Влияние LTV на прибыльность")
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Важные параметры
        param_data = [sample_cpc, sample_conv, sample_repeats]
        param_labels = ['CPC, руб', 'Конверсия, %', 'Сессии']
        
        for i, (data, label) in enumerate(zip(param_data, param_labels)):
            axes[1, 1].scatter(data, sample_profits, alpha=0.3, s=10, label=label)
        
        axes[1, 1].axhline(0, color='red', linestyle='-', alpha=0.5)
        axes[1, 1].set_xlabel("Значения параметров")
        axes[1, 1].set_ylabel("Прибыль, руб")
        axes[1, 1].set_title("Зависимость прибыли от параметров")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Анализ (бюджет: {m_p['budget']:,.0f} руб.)", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_image = f"{project_name}_analysis.png"
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        click.secho(f"\n📊 График сохранен как: {output_image}", fg='yellow')
        plt.close()

if __name__ == '__main__':
    run_psychologist_economy()