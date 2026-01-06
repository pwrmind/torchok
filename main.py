# main_final.py
import torch
import click
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--config', required=True, help='Путь к YAML файлу')
@click.option('--plot', is_flag=True, help='Сгенерировать график')
@click.option('--optimize', is_flag=True, help='Найти оптимальные параметры для достижения 50% успешности')
def run_psychologist_economy(config, plot, optimize):
    config_path = Path(config)
    project_name = config_path.stem 

    with open(config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    m_p, p_p = params['market_params'], params['product_params']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenarios = min(int(m_p['scenarios_count']), 500000)

    # --- РЕАЛИСТИЧНАЯ ГЕНЕРАЦИЯ ---
    
    # CPC: нормальное распределение с реалистичными границами
    cpc_field = torch.normal(m_p['cpc_avg'], m_p['cpc_stdev'], size=(scenarios,), device=device)
    cpc_field = torch.clamp(cpc_field, min=120.0, max=280.0)
    
    # Конверсия: бета-распределение с пиком на 1.5-2%
    alpha, beta = 2.5, 12.0  # Более правдоподобное распределение
    conv_beta = torch.distributions.Beta(alpha, beta).sample((scenarios,)).to(device)
    conv_min, conv_max = p_p['conversion_min'], p_p['conversion_max']
    conv_field = conv_min + conv_beta * (conv_max - conv_min)
    
    # Цена: нормальное распределение с медианой 4000 руб
    price_median = 4000.0
    price_std = 1000.0
    price_field = torch.normal(price_median, price_std, size=(scenarios,), device=device)
    price_field = torch.clamp(price_field, min=p_p['price_range_min'], max=p_p['price_range_max'])
    
    # Повторные сессии: усеченное нормальное (пик на 3-4 сессии)
    repeat_mean = 3.5
    repeat_std = 1.8
    repeat_sessions = torch.normal(repeat_mean, repeat_std, size=(scenarios,), device=device)
    repeat_sessions = torch.clamp(repeat_sessions, min=1.0, max=12.0)
    repeat_sessions = torch.round(repeat_sessions)
    
    # Бюджет с учетом эффективности
    effective_budget = m_p['budget'] * 0.85

    # --- РАСЧЕТ ЭКОНОМИКИ С УЧЕТОМ СКИДОК НА ПАКЕТЫ ---
    with torch.inference_mode():
        clicks = effective_budget / cpc_field
        initial_clients = clicks * conv_field
        initial_clients = torch.clamp(initial_clients, min=0.5)
        
        # Учет пакетов: клиенты с 5+ сессиями получают скидку 15%
        package_mask = repeat_sessions >= 5
        discount_factor = torch.ones_like(repeat_sessions)
        discount_factor[package_mask] = 0.85
        
        # Маржа с учетом скидок на пакеты
        base_margin = price_field * (1 - p_p['tax_rate']) - p_p['base_cogs']
        effective_margin = base_margin * discount_factor
        
        total_sessions = initial_clients * repeat_sessions
        revenue = total_sessions * effective_margin
        profits = revenue - effective_budget
        
        # Метрики
        cac = effective_budget / initial_clients
        ltv = repeat_sessions * effective_margin
        ltv_cac_ratio = ltv / torch.clamp(cac, min=1.0)

    # --- ФИЛЬТРАЦИЯ И АНАЛИЗ ---
    realistic_mask = (
        (conv_field >= 0.006) & 
        (conv_field <= 0.04) & 
        (cpc_field >= 100) & 
        (cpc_field <= 300) &
        (initial_clients >= 1.0)
    )
    
    if not realistic_mask.any():
        click.secho("❌ Нет реалистичных сценариев!", fg='red')
        return
    
    realistic_indices = torch.where(realistic_mask)[0]
    realistic_profits = profits[realistic_indices]
    realistic_count = len(realistic_indices)
    
    # --- НОВЫЙ АЛГОРИТМ ВЫБОРА ОПТИМАЛЬНЫХ ПАРАМЕТРОВ ---
    
    # 1. Находим сценарии в верхних 20% по LTV/CAC (>2.5)
    high_ltv_cac_mask = ltv_cac_ratio[realistic_indices] > 2.5
    if high_ltv_cac_mask.any():
        high_ltv_indices = realistic_indices[high_ltv_cac_mask]
        # Берем медианную цену среди сценариев с высоким LTV/CAC
        median_price_high_ltv = torch.median(price_field[high_ltv_indices])
        
        # Находим сценарий с ценой ближе всего к медианной
        price_diffs = torch.abs(price_field[high_ltv_indices] - median_price_high_ltv)
        closest_idx = torch.argmin(price_diffs)
        best_idx = high_ltv_indices[closest_idx]
    else:
        # Если нет сценариев с LTV/CAC > 2.5, берем лучший по прибыли
        best_idx = realistic_indices[torch.argmax(realistic_profits)]
    
    # --- БАЗОВАЯ СТАТИСТИКА ---
    success_rate = (realistic_profits > 0).sum().item() / realistic_count * 100
    median_profit = torch.median(realistic_profits).item()
    
    # --- ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ---
    opt_price = price_field[best_idx].item()
    opt_profit = profits[best_idx].item()
    opt_conversion = conv_field[best_idx].item() * 100
    opt_cpc = cpc_field[best_idx].item()
    opt_repeat = repeat_sessions[best_idx].item()
    opt_ltv_cac = ltv_cac_ratio[best_idx].item()
    
    # --- АНАЛИЗ ПРИБЫЛЬНЫХ СЦЕНАРИЕВ ---
    profitable_mask = realistic_profits > 0
    if profitable_mask.any():
        profitable_indices = realistic_indices[profitable_mask]
        profitable_prices = price_field[profitable_indices]
        profitable_repeats = repeat_sessions[profitable_indices]
        profitable_ltv_cac = ltv_cac_ratio[profitable_indices]
        
        # Процентили для цен в прибыльных сценариях
        price_25 = torch.quantile(profitable_prices, 0.25).item()
        price_75 = torch.quantile(profitable_prices, 0.75).item()
        
        # Среднее количество сессий в прибыльных сценариях
        avg_repeat_profitable = torch.mean(profitable_repeats.float()).item()
        avg_ltv_cac_profitable = torch.mean(profitable_ltv_cac).item()
        
        # Типичная прибыль (медиана прибыльных сценариев)
        typical_profit = torch.median(realistic_profits[profitable_mask]).item()
    else:
        price_25 = price_75 = avg_repeat_profitable = avg_ltv_cac_profitable = typical_profit = 0
    
    # --- ВЫВОД С УЛУЧШЕННЫМИ РЕКОМЕНДАЦИЯМИ ---
    click.secho(f"\n🧮 РЕАЛЬНАЯ ЭКОНОМИКА ПСИХОЛОГИЧЕСКОЙ ПРАКТИКИ", fg='cyan', bold=True)
    click.secho("=" * 65, fg='cyan')
    
    click.secho(f"\n📈 ОСНОВНЫЕ МЕТРИКИ ({realistic_count:,} реалистичных симуляций):", fg='yellow', bold=True)
    click.echo(f"  Вероятность успеха: {success_rate:.1f}%")
    click.echo(f"  Медианная прибыль: {median_profit:,.0f} руб.")
    click.echo(f"  Рекомендуемая цена: {opt_price:,.0f} руб.")
    click.echo(f"  Ожидаемая прибыль: {opt_profit:,.0f} руб.")
    
    click.secho("\n🎯 КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ ОПТИМАЛЬНОГО СЦЕНАРИЯ:", fg='green', bold=True)
    click.echo(f"  • Конверсия: {opt_conversion:.2f}%")
    click.echo(f"  • Средний CPC: {opt_cpc:.0f} руб.")
    click.echo(f"  • Сессий на клиента: {opt_repeat:.1f}")
    click.echo(f"  • LTV/CAC: {opt_ltv_cac:.2f}")
    
    if profitable_mask.any():
        click.secho("\n📊 АНАЛИЗ ПРИБЫЛЬНЫХ КЕЙСОВ:", fg='blue', bold=True)
        click.echo(f"  Диапазон цен: {price_25:,.0f} - {price_75:,.0f} руб.")
        click.echo(f"  Среднее количество сессий: {avg_repeat_profitable:.1f}")
        click.echo(f"  Средний LTV/CAC: {avg_ltv_cac_profitable:.2f}")
        click.echo(f"  Типичная прибыль: {typical_profit:,.0f} руб.")
    
    # --- РАСШИРЕННЫЕ РЕКОМЕНДАЦИИ ---
    click.secho("\n💡 СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ:", fg='magenta', bold=True)
    
    if success_rate < 30:
        click.secho("  ⚠️  ТЕКУЩАЯ МОДЕЛЬ ВЫСОКОРИСКОВАННА", fg='red')
        click.echo("    1. УВЕЛИЧЬТЕ БЮДЖЕТ до 100,000+ руб./мес")
        click.echo("    2. СНИЖАЙТЕ CPC через:")
        click.echo("       • SEO-оптимизацию сайта")
        click.echo("       • Прямые заявки (сарафанное радио, коллеги)")
        click.echo("       • Таргет в соцсетях на узкие аудитории")
    elif success_rate < 50:
        click.secho("  📊 МОДЕЛЬ ТРЕБУЕТ ОПТИМИЗАЦИИ", fg='yellow')
        click.echo("    1. ОПТИМАЛЬНЫЙ ЦЕНОВОЙ ДИАПАЗОН:")
        click.echo(f"       • Нижняя граница: {max(3000, price_25):,.0f} руб.")
        click.echo(f"       • Верхняя граница: {min(5500, price_75):,.0f} руб.")
        click.echo("    2. ПОВЫШАЙТЕ LTV:")
        click.echo("       • Внедрите пакеты 5+ сессий со скидкой 15%")
        click.echo("       • Система напоминаний о следующей сессии")
        click.echo("       • Программы поддержки между сессиями")
    else:
        click.secho("  ✅ МОДЕЛЬ УСТОЙЧИВАЯ, МОЖНО МАСШТАБИРОВАТЬ", fg='green')
        click.echo("    1. ТЕСТИРУЙТЕ ПОВЫШЕНИЕ ЦЕНЫ:")
        click.echo(f"       • Текущая: {opt_price:,.0f} руб.")
        click.echo(f"       • Тест: {opt_price * 1.1:,.0f} руб. (+10%)")
        click.echo("    2. ИНВЕСТИРУЙТЕ В РАЗВИТИЕ:")
        click.echo("       • Супервизии и обучение")
        click.echo("       • Узкие специализации (пары, дети, trauma)")
    
    # Анализ LTV/CAC
    if avg_ltv_cac_profitable > 0:
        if avg_ltv_cac_profitable < 2.5:
            click.secho(f"\n🔴 КРИТИЧЕСКИЙ ПОКАЗАТЕЛЬ: LTV/CAC = {avg_ltv_cac_profitable:.2f}", fg='red')
            click.echo("   НОРМА ДЛЯ ПСИХОЛОГОВ: 3.0+")
            click.echo("   МЕРЫ ПОВЫШЕНИЯ:")
            click.echo("   1. Увеличивайте средний чек через пакеты")
            click.echo("   2. Повышайте удержание клиентов (retention)")
            click.echo("   3. Снижайте стоимость привлечения (CAC)")
        elif avg_ltv_cac_profitable < 3.5:
            click.secho(f"\n🟡 ПРИЕМЛЕМЫЙ УРОВЕНЬ: LTV/CAC = {avg_ltv_cac_profitable:.2f}", fg='yellow')
            click.echo("   ЦЕЛЬ: довести до 4.0+")
        else:
            click.secho(f"\n✅ ОТЛИЧНЫЙ РЕЗУЛЬТАТ: LTV/CAC = {avg_ltv_cac_profitable:.2f}", fg='green')
    
    # --- ОПТИМИЗАЦИЯ ДЛЯ ДОСТИЖЕНИЯ 50% УСПЕШНОСТИ ---
    if optimize:
        click.secho("\n🔍 ОПТИМИЗАЦИЯ ПАРАМЕТРОВ ДЛЯ 50%+ УСПЕШНОСТИ:", fg='cyan', bold=True)
        
        # Симуляция с разными бюджетами
        budgets = [50000, 75000, 100000, 150000]
        target_success = 50.0
        
        for budget in budgets:
            test_budget = budget * 0.85
            test_profits = (revenue / effective_budget * test_budget) - test_budget
            test_mask = realistic_mask & (test_profits > 0)
            test_success = test_mask.sum().item() / realistic_mask.sum().item() * 100
            
            if test_success >= target_success:
                click.secho(f"  ✅ Бюджет {budget:,.0f} руб. → успешность {test_success:.1f}%", fg='green')
                # Находим оптимальную цену для этого бюджета
                test_realistic_profits = test_profits[realistic_indices]
                test_best_idx = realistic_indices[torch.argmax(test_realistic_profits)]
                test_opt_price = price_field[test_best_idx].item()
                test_opt_profit = test_profits[test_best_idx].item()
                click.echo(f"     • Оптимальная цена: {test_opt_price:,.0f} руб.")
                click.echo(f"     • Ожидаемая прибыль: {test_opt_profit:,.0f} руб.")
                break
        else:
            click.secho(f"  ❌ Даже при 150,000 руб. успешность < {target_success}%", fg='red')
            click.echo("     • Необходимо улучшать конверсию или снижать CPC")
    
    # --- ВИЗУАЛИЗАЦИЯ ---
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        sample_size = min(5000, realistic_count)
        sample_indices = np.random.choice(realistic_count, sample_size, replace=False)
        
        sample_profits = realistic_profits[sample_indices].cpu().numpy()
        sample_prices = price_field[realistic_indices][sample_indices].cpu().numpy()
        sample_repeats = repeat_sessions[realistic_indices][sample_indices].cpu().numpy()
        sample_ltv_cac = ltv_cac_ratio[realistic_indices][sample_indices].cpu().numpy()
        
        # 1. Прибыль vs Цена с цветом по LTV/CAC
        scatter1 = axes[0, 0].scatter(sample_prices, sample_profits, 
                                     c=sample_ltv_cac, cmap='RdYlGn', 
                                     alpha=0.6, s=20, vmin=0, vmax=5)
        axes[0, 0].axhline(0, color='black', linestyle='-', alpha=0.5)
        axes[0, 0].axvline(opt_price, color='blue', linestyle='--', linewidth=2, 
                          label=f'Оптимум: {opt_price:,.0f} руб.')
        axes[0, 0].fill_betweenx([min(sample_profits), max(sample_profits)], 
                                 price_25, price_75, alpha=0.1, color='green',
                                 label=f'Прибыльный диапазон: {price_25:,.0f}-{price_75:,.0f} руб.')
        axes[0, 0].set_xlabel("Цена за сессию (руб)")
        axes[0, 0].set_ylabel("Прибыль (руб)")
        axes[0, 0].set_title(f"Зависимость прибыли от цены (успешность: {success_rate:.1f}%)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 0]).set_label('LTV/CAC')
        
        # 2. Распределение прибыли с акцентом на порог безубыточности
        profit_range = max(abs(min(sample_profits)), abs(max(sample_profits)))
        bins = np.linspace(-profit_range, profit_range, 50)
        
        axes[0, 1].hist(sample_profits[sample_profits > 0], bins=bins[bins > 0], 
                       color='green', alpha=0.6, label='Прибыльные', density=True)
        axes[0, 1].hist(sample_profits[sample_profits <= 0], bins=bins[bins <= 0], 
                       color='red', alpha=0.6, label='Убыточные', density=True)
        axes[0, 1].axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.7)
        axes[0, 1].axvline(median_profit, color='blue', linestyle='--', linewidth=2,
                          label=f'Медиана: {median_profit:,.0f} руб.')
        axes[0, 1].axvline(typical_profit, color='orange', linestyle='--', linewidth=2,
                          label=f'Типичная прибыль: {typical_profit:,.0f} руб.')
        axes[0, 1].set_xlabel("Прибыль (руб)")
        axes[0, 1].set_ylabel("Плотность вероятности")
        axes[0, 1].set_title("Распределение прибыли")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Корреляция LTV/CAC с прибылью
        axes[1, 0].scatter(sample_ltv_cac, sample_profits, alpha=0.5, s=20)
        axes[1, 0].axhline(0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].axvline(2.5, color='red', linestyle='--', alpha=0.7, 
                          label='Минимальный здоровый LTV/CAC')
        axes[1, 0].axvline(3.0, color='green', linestyle='--', alpha=0.7,
                          label='Целевой LTV/CAC')
        axes[1, 0].set_xlabel("LTV/CAC")
        axes[1, 0].set_ylabel("Прибыль (руб)")
        axes[1, 0].set_title("Зависимость прибыли от LTV/CAC")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Тепловая карта: цена vs повторные сессии
        from scipy import stats
        x_bins = np.linspace(min(sample_prices), max(sample_prices), 15)
        y_bins = np.linspace(1, max(sample_repeats), 10)
        
        heatmap, xedges, yedges = np.histogram2d(sample_prices, sample_repeats, 
                                                 bins=[x_bins, y_bins], 
                                                 weights=sample_profits)
        
        im = axes[1, 1].imshow(heatmap.T, origin='lower', aspect='auto', 
                              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                              cmap='RdYlGn', alpha=0.8)
        axes[1, 1].set_xlabel("Цена за сессию (руб)")
        axes[1, 1].set_ylabel("Сессий на клиента")
        axes[1, 1].set_title("Тепловая карта прибыли: цена × LTV")
        axes[1, 1].grid(False)
        plt.colorbar(im, ax=axes[1, 1]).set_label('Суммарная прибыль')
        
        plt.suptitle(f"Детальный анализ психологической практики (бюджет: {m_p['budget']:,.0f} руб.)", 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_image = f"{project_name}_detailed_analysis.png"
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        click.secho(f"\n📊 Детальный отчет сохранен как: {output_image}", fg='yellow')

if __name__ == '__main__':
    run_psychologist_economy()