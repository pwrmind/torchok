# universal_economy_analyzer.py
import torch
import click
import yaml
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--config', required=True, help='Путь к YAML файлу конфигурации')
@click.option('--plot', is_flag=True, help='Сгенерировать график')
@click.option('--optimize', is_flag=True, help='Найти оптимальные параметры')
@click.option('--scenarios', type=int, default=500000, help='Количество сценариев')
def run_universal_economy(config, plot, optimize, scenarios):
    """
    Универсальный экономический симулятор для бизнес-моделей
    
    Поддерживает различные бизнес-модели:
    - Психологическая практика
    - Детейлинг-центр
    - Кофейня
    - SaaS
    - и другие сервисы
    """
    config_path = Path(config)
    project_name = config_path.stem

    with open(config, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    
    m_p, p_p = params['market_params'], params['product_params']
    
    # Определяем бизнес-тип по параметрам конфига
    business_type = identify_business_type(m_p, p_p)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scenarios_count = min(int(m_p.get('scenarios_count', scenarios)), 1000000)

    # --- УНИВЕРСАЛЬНАЯ ГЕНЕРАЦИЯ ПАРАМЕТРОВ ---
    
    # 1. CPC/стоимость лида
    if 'cpc_avg' in m_p:
        cpc_field = generate_cpc(m_p, scenarios_count, device)
    elif 'cost_per_lead_avg' in m_p:
        cpc_field = generate_cost_per_lead(m_p, scenarios_count, device)
    else:
        cpc_field = torch.full((scenarios_count,), 100.0, device=device)
    
    # 2. Конверсия
    if 'conversion_min' in p_p and 'conversion_max' in p_p:
        conv_field = generate_conversion(p_p, scenarios_count, device, business_type)
    elif 'conversion_to_lead_min' in p_p and 'conversion_to_sale_min' in p_p:
        # Для двухэтапной воронки
        conv_field = generate_two_stage_conversion(p_p, scenarios_count, device)
    else:
        conv_field = torch.full((scenarios_count,), 0.02, device=device)
    
    # 3. Цена/чек
    if 'price_range_min' in p_p:
        price_field = generate_price(p_p, scenarios_count, device, business_type)
    elif 'avg_ticket_min' in p_p:
        price_field = generate_ticket(p_p, scenarios_count, device)
    else:
        price_field = torch.full((scenarios_count,), 5000.0, device=device)
    
    # 4. Повторные покупки/LTV
    if 'repeat_sessions_min' in p_p:
        repeat_field = generate_repeat_business(p_p, scenarios_count, device, business_type)
    elif 'repeat_purchases_min' in p_p:
        repeat_field = generate_repeat_purchases(p_p, scenarios_count, device)
    elif 'subscription_months_min' in p_p:
        repeat_field = generate_subscription(p_p, scenarios_count, device)
    else:
        repeat_field = torch.ones((scenarios_count,), device=device)
    
    # 5. Бюджет
    budget = m_p['budget']
    effective_budget = budget * m_p.get('budget_efficiency', 0.85)

    # --- УНИВЕРСАЛЬНЫЙ РАСЧЕТ ЭКОНОМИКИ ---
    with torch.inference_mode():
        # Лиды/трафик
        if 'cpc_avg' in m_p:
            traffic = effective_budget / cpc_field
        else:
            traffic = effective_budget / cpc_field  # cost_per_lead
        
        # Конверсия в продажи
        sales = traffic * conv_field
        sales = torch.clamp(sales, min=0.5)
        
        # Общий объем (с учетом повторных)
        total_volume = sales * repeat_field
        
        # Маржа
        tax_rate = p_p.get('tax_rate', 0.06)
        base_cogs = p_p.get('base_cogs', 0.0)
        
        # Учитываем скидки на опт/пакеты для разных бизнес-моделей
        if business_type in ['detailing', 'psychology'] and 'repeat_sessions_min' in p_p:
            # Скидка за пакеты услуг
            package_discount = torch.where(
                repeat_field >= 3,
                1.0 - 0.1 * (repeat_field // 3),
                torch.ones_like(repeat_field)
            )
            unit_margin = (price_field * (1 - tax_rate) - base_cogs) * package_discount
        elif business_type == 'subscription':
            # Для подписок - другая модель
            unit_margin = price_field * (1 - tax_rate) - base_cogs
        else:
            unit_margin = price_field * (1 - tax_rate) - base_cogs
        
        # Прибыль
        revenue = total_volume * unit_margin
        profits = revenue - effective_budget
        
        # Метрики
        cac = effective_budget / sales
        ltv = repeat_field * unit_margin
        ltv_cac_ratio = ltv / torch.clamp(cac, min=1.0)
        
        # ROMI (Return on Marketing Investment)
        romi = (revenue - effective_budget) / effective_budget

    # --- АНАЛИЗ И ВЫВОД ---
    realistic_mask = filter_realistic_scenarios(
        conv_field, cpc_field, sales, profits, 
        business_type, p_p
    )
    
    if not realistic_mask.any():
        click.secho("❌ Нет реалистичных сценариев!", fg='red')
        return
    
    # Статистика
    realistic_profits = profits[realistic_mask]
    realistic_count = realistic_mask.sum().item()
    success_rate = (realistic_profits > 0).sum().item() / realistic_count * 100
    median_profit = torch.median(realistic_profits).item()
    
    # Находим оптимальный сценарий
    optimal_idx = find_optimal_scenario(
        profits, ltv_cac_ratio, price_field, 
        realistic_mask, business_type
    )
    
    # --- ВЫВОД РЕЗУЛЬТАТОВ ---
    display_results(
        project_name, business_type, realistic_count, success_rate,
        median_profit, optimal_idx, price_field, profits, conv_field,
        cpc_field, repeat_field, ltv_cac_ratio, unit_margin, budget,
        m_p, p_p
    )
    
    # Рекомендации для конкретного бизнес-типа
    display_recommendations(
        business_type, success_rate, ltv_cac_ratio[optimal_idx].item(),
        price_field[optimal_idx].item(), realistic_profits, 
        profits[optimal_idx].item(), budget
    )
    
    # --- ВИЗУАЛИЗАЦИЯ ---
    if plot:
        create_visualization(
            project_name, business_type, realistic_mask,
            price_field, profits, conv_field, cpc_field,
            repeat_field, ltv_cac_ratio, budget
        )
    
    # --- ОПТИМИЗАЦИЯ ---
    if optimize:
        run_optimization(
            business_type, budget, revenue, effective_budget,
            realistic_mask, price_field, conv_field, cpc_field,
            repeat_field, p_p
        )

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def identify_business_type(market_params, product_params):
    """Определяем тип бизнеса по параметрам"""
    price_min = product_params.get('price_range_min', 0)
    price_max = product_params.get('price_range_max', 0)
    base_cogs = product_params.get('base_cogs', 0)
    
    if price_max > 50000:
        return 'detailing'
    elif price_max > 10000:
        return 'premium_service'
    elif 'subscription_months_min' in product_params:
        return 'subscription'
    elif base_cogs > 5000:
        return 'high_cogs_service'
    elif price_max < 5000:
        return 'low_ticket_service'
    else:
        return 'psychology'

def generate_cpc(market_params, scenarios, device):
    """Генерация CPC"""
    cpc_avg = market_params['cpc_avg']
    cpc_std = market_params['cpc_stdev']
    cpc_low = max(50.0, cpc_avg - 2*cpc_std)
    cpc_high = min(500.0, cpc_avg + 2*cpc_std)
    
    cpc = torch.normal(cpc_avg, cpc_std, size=(scenarios,), device=device)
    return torch.clamp(cpc, min=cpc_low, max=cpc_high)

def generate_conversion(product_params, scenarios, device, business_type):
    """Генерация конверсии в зависимости от типа бизнеса"""
    conv_min = product_params['conversion_min']
    conv_max = product_params['conversion_max']
    
    if business_type in ['detailing', 'premium_service']:
        # Для премиальных услуг - пик на нижних значениях
        alpha, beta = 1.5, 6.0
        conv_beta = torch.distributions.Beta(alpha, beta).sample((scenarios,)).to(device)
    else:
        # Для остальных - более равномерное распределение
        alpha, beta = 2.0, 4.0
        conv_beta = torch.distributions.Beta(alpha, beta).sample((scenarios,)).to(device)
    
    conv = conv_min + conv_beta * (conv_max - conv_min)
    return torch.clamp(conv, min=conv_min*0.8, max=conv_max*1.2)

def generate_price(product_params, scenarios, device, business_type):
    """Генерация цены"""
    price_min = product_params['price_range_min']
    price_max = product_params['price_range_max']
    
    if business_type == 'detailing':
        # Для детейлинга - пик на 40-60k
        price_mode = (price_min + price_max) / 2
        price_std = (price_max - price_min) / 4
    elif business_type == 'psychology':
        # Для психологов - пик на 3500-4500
        price_mode = 4000.0
        price_std = 1000.0
    else:
        price_mode = (price_min + price_max) / 2
        price_std = (price_max - price_min) / 6
    
    price = torch.normal(price_mode, price_std, size=(scenarios,), device=device)
    return torch.clamp(price, min=price_min, max=price_max)

def generate_repeat_business(product_params, scenarios, device, business_type):
    """Генерация повторных покупок"""
    repeat_min = product_params['repeat_sessions_min']
    repeat_max = product_params['repeat_sessions_max']
    
    if business_type == 'detailing':
        # Для детейлинга - большинство 1-2 посещения, некоторые постоянные
        geom_p = 0.3
        repeat = torch.distributions.Geometric(geom_p).sample((scenarios,)).to(device) + 1
    elif business_type == 'psychology':
        # Для психологов - нормальное распределение
        repeat_mean = (repeat_min + repeat_max) / 2
        repeat_std = (repeat_max - repeat_min) / 6
        repeat = torch.normal(repeat_mean, repeat_std, size=(scenarios,), device=device)
    else:
        # Равномерное распределение
        repeat = torch.distributions.Uniform(repeat_min, repeat_max).sample((scenarios,)).to(device)
    
    repeat = torch.clamp(repeat, min=repeat_min, max=repeat_max)
    return torch.round(repeat)

def filter_realistic_scenarios(conv_field, cpc_field, sales, profits, business_type, product_params):
    """Фильтрация реалистичных сценариев"""
    if business_type == 'detailing':
        return (
            (conv_field >= 0.005) & (conv_field <= 0.05) &
            (cpc_field >= 80) & (cpc_field <= 400) &
            (sales >= 1.0) &
            (profits > -200000) & (profits < 500000)
        )
    elif business_type == 'psychology':
        return (
            (conv_field >= 0.005) & (conv_field <= 0.05) &
            (cpc_field >= 80) & (cpc_field <= 400) &
            (sales >= 0.5) &
            (profits > -100000) & (profits < 300000)
        )
    else:
        return (
            (conv_field >= 0.005) & (conv_field <= 0.1) &
            (cpc_field >= 50) & (cpc_field <= 500) &
            (sales >= 0.5) &
            torch.isfinite(profits)
        )

def find_optimal_scenario(profits, ltv_cac_ratio, price_field, realistic_mask, business_type):
    """Поиск оптимального сценария"""
    realistic_indices = torch.where(realistic_mask)[0]
    realistic_profits = profits[realistic_indices]
    
    # Для премиальных услуг приоритет LTV/CAC
    if business_type in ['detailing', 'subscription', 'premium_service']:
        high_ltv_mask = ltv_cac_ratio[realistic_indices] > 2.5
        if high_ltv_mask.any():
            high_ltv_indices = realistic_indices[high_ltv_mask]
            # Берем медианную цену среди сценариев с высоким LTV/CAC
            median_price = torch.median(price_field[high_ltv_indices])
            price_diffs = torch.abs(price_field[high_ltv_indices] - median_price)
            closest_idx = torch.argmin(price_diffs)
            return high_ltv_indices[closest_idx]
    
    # Для остальных - максимальная прибыль в реалистичных
    return realistic_indices[torch.argmax(realistic_profits)]

def display_results(project_name, business_type, realistic_count, success_rate,
                    median_profit, optimal_idx, price_field, profits, conv_field,
                    cpc_field, repeat_field, ltv_cac_ratio, unit_margin, budget,
                    market_params, product_params):
    """Отображение результатов"""
    
    business_names = {
        'detailing': 'Детейлинг-центр',
        'psychology': 'Психологическая практика',
        'subscription': 'Подписка/SaaS',
        'premium_service': 'Премиум-сервис',
        'high_cogs_service': 'Сервис с высокими COGS',
        'low_ticket_service': 'Сервис с низким чеком'
    }
    
    business_name = business_names.get(business_type, 'Бизнес-проект')
    
    click.secho(f"\n🚀 ЭКОНОМИЧЕСКИЙ АНАЛИЗ: {business_name}", fg='cyan', bold=True)
    click.secho("=" * 65, fg='cyan')
    
    click.secho(f"\n📊 БАЗОВЫЕ ПАРАМЕТРЫ:", fg='yellow', bold=True)
    click.echo(f"  Проект: {project_name}")
    click.echo(f"  Бюджет на маркетинг: {budget:,.0f} руб.")
    click.echo(f"  Анализировано сценариев: {realistic_count:,}")
    
    click.secho(f"\n📈 КЛЮЧЕВЫЕ МЕТРИКИ:", fg='green', bold=True)
    click.echo(f"  Вероятность успеха: {success_rate:.1f}%")
    click.echo(f"  Медианная прибыль: {median_profit:,.0f} руб.")
    
    click.secho(f"\n🎯 ОПТИМАЛЬНЫЕ НАСТРОЙКИ:", fg='magenta', bold=True)
    click.echo(f"  Средний чек: {price_field[optimal_idx].item():,.0f} руб.")
    click.echo(f"  Ожидаемая прибыль: {profits[optimal_idx].item():,.0f} руб.")
    click.echo(f"  Конверсия: {conv_field[optimal_idx].item()*100:.2f}%")
    click.echo(f"  CPC: {cpc_field[optimal_idx].item():.0f} руб.")
    
    if 'repeat_sessions_min' in product_params:
        click.echo(f"  Повторные продажи: {repeat_field[optimal_idx].item():.1f}")
    
    click.echo(f"  LTV/CAC: {ltv_cac_ratio[optimal_idx].item():.2f}")
    click.echo(f"  Маржа на единицу: {unit_margin[optimal_idx].item():,.0f} руб.")

def display_recommendations(business_type, success_rate, ltv_cac, 
                           optimal_price, realistic_profits, optimal_profit, budget):
    """Вывод рекомендаций в зависимости от типа бизнеса"""
    
    click.secho(f"\n💡 СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ:", fg='cyan', bold=True)
    
    if business_type == 'detailing':
        click.secho("  🚗 ДЕТЕЙЛИНГ-ЦЕНТР:", fg='blue', bold=True)
        if success_rate < 40:
            click.secho("    ⚠️  РИСК СРЕДНИЙ", fg='yellow')
            click.echo("    1. Фокус на повторных клиентах (программы лояльности)")
            click.echo("    2. Оптимизируйте портфель услуг (керамика vs полировка)")
            click.echo(f"    3. Целевая цена: {optimal_price*0.9:,.0f}-{optimal_price*1.1:,.0f} руб.")
        else:
            click.secho("    ✅ УСТОЙЧИВАЯ МОДЕЛЬ", fg='green')
            click.echo("    1. Расширяйте услуги (тонирование, бронирование)")
            click.echo("    2. Инвестируйте в оборудование и обучение мастеров")
            click.echo("    3. Внедряйте CRM для удержания клиентов")
    
    elif business_type == 'psychology':
        click.secho("  🧠 ПСИХОЛОГИЧЕСКАЯ ПРАКТИКА:", fg='magenta', bold=True)
        if success_rate < 35:
            click.secho("    ⚠️  ВЫСОКИЙ РИСК", fg='yellow')
            click.echo("    1. Увеличьте бюджет до 100,000+ руб.")
            click.echo("    2. Снижайте CPC через контент-маркетинг")
            click.echo("    3. Повышайте доверие через отзывы и кейсы")
        else:
            click.secho("    ✅ ПРИЕМЛЕМЫЙ РИСК", fg='green')
            click.echo("    1. Оптимизируйте воронку продаж")
            click.echo("    2. Внедрите пакеты сессий со скидкой")
            click.echo("    3. Развивайте узкую специализацию")
    
    else:
        if success_rate < 30:
            click.secho("    🔴 ВЫСОКИЙ РИСК", fg='red')
            click.echo(f"    1. Увеличьте бюджет (минимум {budget*1.5:,.0f} руб.)")
            click.echo("    2. Снижайте стоимость привлечения")
            click.echo("    3. Тестируйте разные ценовые точки")
        elif success_rate < 60:
            click.secho("    🟡 СРЕДНИЙ РИСК", fg='yellow')
            click.echo("    1. Оптимизируйте LTV через удержание")
            click.echo("    2. Улучшайте конверсию на сайте")
            click.echo("    3. Тестируйте новые каналы привлечения")
        else:
            click.secho("    🟢 НИЗКИЙ РИСК", fg='green')
            click.echo("    1. Масштабируйте успешные каналы")
            click.echo("    2. Инвестируйте в автоматизацию")
            click.echo("    3. Расширяйте продуктовую линейку")
    
    # Рекомендации по LTV/CAC
    if ltv_cac > 0:
        if ltv_cac < 2.5:
            click.secho(f"\n⚠️  LTV/CAC {ltv_cac:.1f} ниже нормы (3.0+)", fg='yellow')
            click.echo("   Меры по улучшению:")
            click.echo("   1. Увеличивайте средний чек")
            click.echo("   2. Повышайте retention клиентов")
            click.echo("   3. Снижайте стоимость привлечения")
        elif ltv_cac < 4.0:
            click.secho(f"\n✅ LTV/CAC {ltv_cac:.1f} на хорошем уровне", fg='green')
        else:
            click.secho(f"\n🎯 ОТЛИЧНЫЙ LTV/CAC {ltv_cac:.1f}", fg='cyan')

def create_visualization(project_name, business_type, realistic_mask,
                        price_field, profits, conv_field, cpc_field,
                        repeat_field, ltv_cac_ratio, budget):
    """Создание визуализации"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    realistic_indices = torch.where(realistic_mask)[0]
    sample_size = min(5000, len(realistic_indices))
    sample_indices = np.random.choice(len(realistic_indices), sample_size, replace=False)
    
    sample_profits = profits[realistic_indices][sample_indices].cpu().numpy()
    sample_prices = price_field[realistic_indices][sample_indices].cpu().numpy()
    sample_ltv_cac = ltv_cac_ratio[realistic_indices][sample_indices].cpu().numpy()
    sample_repeats = repeat_field[realistic_indices][sample_indices].cpu().numpy()
    
    # 1. Основной график прибыли
    scatter = axes[0, 0].scatter(sample_prices, sample_profits, 
                                c=sample_ltv_cac, cmap='RdYlGn',
                                alpha=0.6, s=20, vmin=0, vmax=5)
    axes[0, 0].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[0, 0].set_xlabel("Цена/чек (руб)")
    axes[0, 0].set_ylabel("Прибыль (руб)")
    axes[0, 0].set_title("Прибыль vs Цена")
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0]).set_label('LTV/CAC')
    
    # 2. Распределение прибыли
    axes[0, 1].hist(sample_profits, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='-', linewidth=2)
    axes[0, 1].set_xlabel("Прибыль (руб)")
    axes[0, 1].set_ylabel("Частота")
    axes[0, 1].set_title("Распределение прибыли")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Влияние повторных продаж
    if torch.max(repeat_field) > 1:
        axes[1, 0].scatter(sample_repeats, sample_profits, alpha=0.5, s=20)
        axes[1, 0].axhline(0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_xlabel("Повторные продажи")
        axes[1, 0].set_ylabel("Прибыль (руб)")
        axes[1, 0].set_title("Влияние LTV на прибыль")
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Корреляция LTV/CAC с прибылью
    axes[1, 1].scatter(sample_ltv_cac, sample_profits, alpha=0.5, s=20)
    axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].axvline(2.5, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(3.0, color='green', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel("LTV/CAC")
    axes[1, 1].set_ylabel("Прибыль (руб)")
    axes[1, 1].set_title("Зависимость прибыли от LTV/CAC")
    axes[1, 1].grid(True, alpha=0.3)
    
    business_titles = {
        'detailing': 'Детейлинг-центр',
        'psychology': 'Психологическая практика',
        'subscription': 'SaaS/Подписка'
    }
    
    title = business_titles.get(business_type, 'Бизнес-анализ')
    plt.suptitle(f"{title} | Бюджет: {budget:,.0f} руб.", 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_image = f"{project_name}_{business_type}_analysis.png"
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    click.secho(f"\n📊 График сохранен как: {output_image}", fg='yellow')

def run_optimization(business_type, budget, revenue, effective_budget,
                    realistic_mask, price_field, conv_field, cpc_field,
                    repeat_field, product_params):
    """Оптимизация параметров"""
    
    click.secho(f"\n🔍 АНАЛИЗ ОПТИМИЗАЦИИ:", fg='cyan', bold=True)
    
    # Тестируем разные бюджеты
    budgets = [budget * 0.5, budget, budget * 1.5, budget * 2]
    
    for test_budget in budgets:
        test_effective = test_budget * 0.85
        test_profits = (revenue / effective_budget * test_effective) - test_effective
        test_success = (test_profits[realistic_mask] > 0).sum().item() / realistic_mask.sum().item() * 100
        
        if test_success >= 50:
            click.secho(f"  ✅ Бюджет {test_budget:,.0f} руб. → {test_success:.1f}% успешности", fg='green')
            
            # Находим оптимальную цену для этого бюджета
            test_best_idx = torch.argmax(test_profits[realistic_mask])
            realistic_indices = torch.where(realistic_mask)[0]
            best_idx = realistic_indices[test_best_idx]
            
            click.echo(f"     • Оптимальная цена: {price_field[best_idx].item():,.0f} руб.")
            click.echo(f"     • Конверсия: {conv_field[best_idx].item()*100:.2f}%")
            click.echo(f"     • CPC: {cpc_field[best_idx].item():.0f} руб.")
            if torch.max(repeat_field) > 1:
                click.echo(f"     • Повторные: {repeat_field[best_idx].item():.1f}")
            break
    else:
        click.secho(f"  ⚠️  Даже при {budgets[-1]:,.0f} руб. успешность < 50%", fg='yellow')
        click.echo("     • Необходимо улучшать конверсию или снижать стоимость привлечения")

# === ФУНКЦИИ ДЛЯ ДРУГИХ БИЗНЕС-МОДЕЛЕЙ ===

def generate_cost_per_lead(market_params, scenarios, device):
    """Для бизнесов с оплатой за лид"""
    cost_avg = market_params['cost_per_lead_avg']
    cost_std = market_params.get('cost_per_lead_std', cost_avg * 0.3)
    cost = torch.normal(cost_avg, cost_std, size=(scenarios,), device=device)
    return torch.clamp(cost, min=cost_avg*0.5, max=cost_avg*1.5)

def generate_ticket(product_params, scenarios, device):
    """Генерация среднего чека"""
    ticket_min = product_params['avg_ticket_min']
    ticket_max = product_params['avg_ticket_max']
    ticket_mode = product_params.get('avg_ticket_mode', (ticket_min + ticket_max) / 2)
    
    if 'ticket_std' in product_params:
        ticket_std = product_params['ticket_std']
    else:
        ticket_std = (ticket_max - ticket_min) / 6
    
    ticket = torch.normal(ticket_mode, ticket_std, size=(scenarios,), device=device)
    return torch.clamp(ticket, min=ticket_min, max=ticket_max)

def generate_subscription(product_params, scenarios, device):
    """Генерация длительности подписки"""
    months_min = product_params['subscription_months_min']
    months_max = product_params['subscription_months_max']
    
    # Экспоненциальное распределение для подписок (большинство короткие)
    lambda_param = 0.5
    subscription = torch.distributions.Exponential(lambda_param).sample((scenarios,)).to(device)
    subscription = subscription * (months_max - months_min) + months_min
    return torch.clamp(subscription, min=months_min, max=months_max)

def generate_two_stage_conversion(product_params, scenarios, device):
    """Для двухэтапной воронки (клик → лид → продажа)"""
    lead_conv = torch.distributions.Uniform(
        product_params['conversion_to_lead_min'],
        product_params['conversion_to_lead_max']
    ).sample((scenarios,)).to(device)
    
    sale_conv = torch.distributions.Uniform(
        product_params['conversion_to_sale_min'],
        product_params['conversion_to_sale_max']
    ).sample((scenarios,)).to(device)
    
    return lead_conv * sale_conv

if __name__ == '__main__':
    run_universal_economy()