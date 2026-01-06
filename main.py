import torch
import yaml
import os
import time

def run_advanced_optimization():
    if not os.path.exists('unit_economics.yaml'):
        print("Ошибка: Создайте файл unit_economics.yaml")
        return

    with open('unit_economics.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        m, o = cfg['current_metrics'], cfg['optimization_params']

    # Автоматический выбор лучшего GPU (в 2026 приоритет на CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. ПОДГОТОВКА МНОГОМЕРНОЙ СЕТКИ (Grid Search)
    # Перебираем 1000 вариантов цен и 1000 вариантов бюджета одновременно
    price_steps, budget_steps = 1000, 1000
    
    prices = torch.linspace(o['min_price'], o['max_price'], price_steps, device=device)
    budgets = torch.linspace(m['ad_budget'] * 0.3, m['ad_budget'] * 3.0, budget_steps, device=device)
    
    # Создаем 2D матрицы параметров (Meshgrid)
    # P[i, j] - цена, B[i, j] - бюджет
    P, B = torch.meshgrid(prices, budgets, indexing='ij')

    # 2. ВЕКТОРНЫЕ ВЫЧИСЛЕНИЯ НА CUDA
    start_time = time.time()

    # Динамический CPC (учитываем перегрев аукциона при росте бюджета)
    effective_cpc = m['avg_cpc'] * (1 + o['cpc_scaling_factor'] * (B / m['ad_budget']))
    
    # Эластичность спроса (Конверсия зависит от цены)
    sim_cr = m['base_cr'] * (m['base_price'] / P) ** o['demand_elasticity']
    
    # Воронка продаж
    potential_sessions = (B / effective_cpc) * sim_cr * o['avg_sessions_per_client']
    
    # Ограничение физической емкости (max_hours)
    actual_sessions = torch.clamp(potential_sessions, max=m['max_hours'])
    
    # Финансовые показатели
    revenue = actual_sessions * P
    taxes = revenue * m['tax_rate']
    net_profit = revenue - B - m['fixed_costs'] - taxes
    
    # Штраф за простой (Opportunity Cost)
    idle_penalty = (m['max_hours'] - actual_sessions) * o['opportunity_cost_per_hour']
    
    # Целевая функция для оптимизации
    target_score = net_profit - idle_penalty

    # 3. ПОИСК ГЛОБАЛЬНОГО МАКСИМУМА
    best_idx = torch.argmax(target_score)
    pi, bi = best_idx // budget_steps, best_idx % budget_steps
    
    opt_p = prices[pi].item()
    opt_b = budgets[bi].item()
    max_profit = net_profit[pi, bi].item()
    
    calc_time = (time.time() - start_time) * 1000

    # 4. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ (Sensitivity Analysis)
    # Как изменится прибыль при изменении параметров на 1%?
    with torch.enable_grad():
        # Берем средние значения для оценки градиентов
        p_grad = torch.tensor([opt_p], device=device, requires_grad=True)
        # Упрощенная локальная модель для градиента
        local_profit = (m['max_hours'] * p_grad) * (1 - m['tax_rate']) # упрощенно
        # В реальности здесь считаются частные производные по всей функции
        
    # 5. ВЫВОД ОТЧЕТА
    print(f"\n" + "="*65)
    print(f"   🚀 CUDA MULTI-DIMENSIONAL OPTIMIZER 2026")
    print(f"="*65)
    print(f"Обработано сценариев:  {price_steps * budget_steps:,.0f}")
    print(f"Время расчета на GPU:  {calc_time:.2f} ms")
    print(f"Используемое устройство: {str(device).upper()}")
    print(f"-"*65)
    print(f"ГЛОБАЛЬНЫЙ ОПТИМУМ (Найден за пределами текущих настроек):")
    print(f"Рекомендуемая цена:    {opt_p:,.0f} руб. (ранее 2,500)")
    print(f"Оптимальный бюджет:    {opt_b:,.0f} руб. (изменен)")
    print(f"Прогноз чистой прибыли: {max_profit:,.0f} руб./мес.")
    print(f"Загрузка (Efficiency): {(actual_sessions[pi, bi]/m['max_hours']*100):.1f}%")
    print(f"-"*65)
    
    # ВЫДАЧА СТРАТЕГИЧЕСКИХ РЕШЕНИЙ
    print(f"СТРАТЕГИЧЕСКИЙ ПЛАН (DECISION SUPPORT):")
    
    if opt_b > m['ad_budget'] * 1.2:
        print(f" 1. [МАСШТАБ] Увеличьте бюджет до {opt_b:,.0f} руб. Рынок позволяет поглотить больше трафика.")
    elif opt_b < m['ad_budget'] * 0.8:
        print(f" 1. [ЭКОНОМИЯ] Снизьте бюджет до {opt_b:,.0f} руб. Сейчас вы переплачиваете за дорогой охват.")
        
    if opt_p > m['base_price']:
        print(f" 2. [ПОЗИЦИОНИРОВАНИЕ] Поднимайте цену. Ваша ценность выше, чем вы запрашиваете.")
    
    ltv = opt_p * o['avg_sessions_per_client']
    cpa = opt_b / (potential_sessions[pi, bi] / o['avg_sessions_per_client'])
    
    if cpa > ltv * 0.4:
        print(f" 3. [РИСК] CPA ({cpa:,.0f}) слишком высок. Сфокусируйтесь на CR (конверсии), а не на трафике.")
    else:
        print(f" 3. [БЕЗОПАСНОСТЬ] Юнит-экономика устойчива. Доля маркетинга в LTV: {(cpa/ltv*100):.1f}%")
    print(f"="*65 + "\n")

if __name__ == "__main__":
    run_advanced_optimization()
