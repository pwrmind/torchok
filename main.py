import torch
import yaml
import os

def generate_expert_advice(m, opt_p, max_p, vol, max_sessions, cpa, ltv):
    """Интеллектуальная интерпретация результатов расчетов."""
    advice = []
    utilization = vol / max_sessions
    
    # 1. Анализ пропускной способности (Capacity)
    if utilization >= 0.98:
        advice.append("[!] ПЕРЕГРУЗКА: График заполнен на 100%. Рост возможен только через "
                      "повышение цены или сокращение сессии до 45-50 минут.")
    elif utilization < 0.50:
        advice.append("[?] НЕДОЗАГРУЗКА: Простой более 50%. Математический оптимум завышен. "
                      "Рекомендуется снизить цену на 15-20% для стимуляции объема.")

    # 2. Маркетинговая эффективность (CPA vs LTV)
    if cpa > (ltv * 0.45):
        advice.append("[!] РИСК МАРКЕТИНГА: CPA превышает 45% от LTV. Вы работаете на рекламную площадку. "
                      "Необходимо срочно улучшать конверсию сайта или искать органику.")
    elif cpa < (ltv * 0.15):
        advice.append("[+] ЭФФЕКТИВНОСТЬ: CPA ниже 15% LTV. Реклама сверхприбыльна. "
                      "Рекомендуется увеличить бюджет на 30-50%.")

    # 3. Финансовая устойчивость
    profit_buffer = max_p / m['fixed_costs'] if m['fixed_costs'] > 0 else 0
    if profit_buffer < 1.2:
        advice.append("[X] НЕУСТОЙЧИВОСТЬ: Прибыль едва покрывает расходы (буфер < 20%). "
                      "Любое колебание CPC приведет к убыткам.")
    elif profit_buffer > 4.0:
        advice.append("[+++] ЭКСПАНСИЯ: Модель крайне прибыльна. Пора нанимать ассистента "
                      "или масштабировать личный бренд.")

    return advice

def run_optimization():
    # Загрузка конфигурации
    if not os.path.exists('unit_economics.yaml'):
        print("Ошибка: Создайте файл unit_economics.yaml")
        return

    with open('unit_economics.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        m = cfg['current_metrics']
        o = cfg['optimization_params']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Константы на устройство
    cpc = torch.tensor(m['avg_cpc'], device=device)
    base_cr = torch.tensor(m['base_cr'], device=device)
    base_p = torch.tensor(m['base_price'], device=device)
    budget = torch.tensor(m['ad_budget'], device=device)
    fixed_costs = torch.tensor(m['fixed_costs'], device=device)
    
    elasticity = o['demand_elasticity']
    ltv_factor = o.get('avg_sessions_per_client', 1.0)
    opp_cost = torch.tensor(o.get('opportunity_cost_per_hour', 0.0), device=device)
    
    max_sessions = m['max_hours'] / m['session_duration']
    max_sessions_t = torch.tensor(max_sessions, device=device)

    # Генерация цен на GPU
    prices = torch.linspace(o['min_price'], o['max_price'], steps=o['steps'], device=device)

    # --- Векторизованные вычисления на CUDA ---
    
    # 1. Спрос с учетом эластичности
    sim_cr = base_cr * (base_p / prices) ** elasticity
    
    # 2. Воронка продаж с учетом LTV
    potential_clients = (budget / cpc) * sim_cr
    total_sessions_demanded = potential_clients * ltv_factor
    
    # 3. Реализованный объем с ограничением по часам
    actual_sessions = torch.clamp(total_sessions_demanded, max=max_sessions_t)
    
    # 4. Экономика со штрафом за простой
    revenue = actual_sessions * prices
    idle_penalty = (max_sessions_t - actual_sessions) * opp_cost
    
    # Оптимизируем прибыль за вычетом штрафа за пустые часы
    target_profit = revenue - budget - fixed_costs - idle_penalty

    # --- Получение результатов ---
    best_idx = torch.argmax(target_profit)
    opt_p = prices[best_idx].item()
    
    # Фактические показатели в точке оптимума
    final_vol = actual_sessions[best_idx].item()
    final_profit = (revenue[best_idx] - budget - fixed_costs).item()
    final_cpa = (m['ad_budget'] / (final_vol / ltv_factor)) if final_vol > 0 else 0
    final_ltv = opt_p * ltv_factor

    # --- Отчет ---
    print(f"\n" + "="*65)
    print(f"   ЭКСПЕРТНЫЙ АНАЛИЗ ЮНИТ-ЭКОНОМИКИ (CUDA 2026)")
    print(f"="*65)
    print(f"Оптимальная цена:      {opt_p:,.0f} руб./сессия")
    print(f"Чистая прибыль:        {final_profit:,.0f} руб./мес.")
    print(f"Загрузка:              {final_vol:,.1f} сессий ({final_vol/max_sessions*100:,.1f}%)")
    print(f"Стоимость привлечения: {final_cpa:,.0f} руб. (CPA)")
    print(f"Выручка с клиента:     {final_ltv:,.0f} руб. (LTV)")
    print(f"Расчетное устройство:  {str(device).upper()}")
    print(f"-"*65)
    print(f"АНАЛИТИЧЕСКИЕ ВЫВОДЫ:")
    
    advices = generate_expert_advice(m, opt_p, final_profit, final_vol, max_sessions, final_cpa, final_ltv)
    for i, msg in enumerate(advices, 1):
        print(f" {i}. {msg}")
    print(f"="*65 + "\n")

if __name__ == "__main__":
    run_optimization()
