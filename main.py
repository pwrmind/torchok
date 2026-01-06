import torch
import yaml
import os
import math

def generate_expert_advice(m, opt_p, profit, vol, max_sessions, cpa, ltv):
    """Интеллектуальная интерпретация результатов расчетов, дающая конкретные рекомендации."""
    advice = []
    utilization = vol / max_sessions
    profit_margin = (profit / (profit + m['fixed_costs'] + m['ad_budget'])) if (profit + m['fixed_costs'] + m['ad_budget']) > 0 else 0

    # 1. СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ (Приоритет 1)
    if profit <= 0:
        advice.append("[!!!] КРИЗИС: Бизнес генерирует убыток. Необходим срочный пересмотр модели. Сократите расходы или поднимите цены немедленно.")
    elif profit_margin < 0.1:
        advice.append("[!] НЕСТАБИЛЬНОСТЬ: Маржа менее 10%. Модель крайне чувствительна к колебаниям рынка (CPC/CR). Срочно ищите пути повышения эффективности.")
    elif profit_margin > 0.4:
        advice.append("[+++] МАСШТАБИРОВАНИЕ: Маржа выше 40%. Модель сверхприбыльна. Готовьтесь к найму команды или открытию новых направлений.")

    # 2. ОПЕРАЦИОННЫЕ РЕКОМЕНДАЦИИ (Приоритет 2)
    if utilization >= 0.95:
        # Предлагаем не просто "поднять цену", а конкретные шаги
        price_increase_needed = m['max_price'] - opt_p
        advice.append(f"[ОПЕРАЦИОНКА] ПЕРЕГРУЗКА: График заполнен на 95%+. Для роста необходимо повышение цены (потенциал на +{price_increase_needed:,.0f} руб.) или сокращение длительности сессии.")
    elif utilization < 0.50:
        # Предлагаем стимуляцию спроса
        advice.append(f"[ОПЕРАЦИОНКА] НЕДОЗАГРУЗКА: Простой более 50%. Рекомендуется снизить цену на 15-20% для стимуляции объема, либо пересмотреть рекламную стратегию.")

    # 3. МАРКЕТИНГОВЫЕ РЕКОМЕНДАЦИИ (Приоритет 3)
    if cpa > ltv * 0.45:
         # Предлагаем 3 варианта действий
        advice.append(f"[МАРКЕТИНГ] РИСК: CPA ({cpa:,.0f} руб.) превышает 45% LTV ({ltv:,.0f} руб.). Необходимо срочно улучшать конверсию сайта, снижать CPC через A/B тесты креативов или искать органический трафик.")
    elif cpa < ltv * 0.15:
        advice.append("[МАРКЕТИНГ] ЭФФЕКТИВНОСТЬ: Реклама сверхприбыльна. Рекомендуется увеличить бюджет на 30-50% для масштабирования успеха.")

    return advice

# Вспомогательные функции (run_optimization) остаются без изменений
def run_optimization():
    # ... (код функции run_optimization остается как в предыдущем ответе) ...
    # ... (не забудьте импортировать generate_expert_advice в начало run_optimization)
    # ... 
    with open('unit_economics.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        m = cfg['current_metrics']
        o = cfg['optimization_params']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    budget = torch.tensor(float(m['ad_budget']), device=device)
    base_cpc = torch.tensor(float(m['avg_cpc']), device=device)
    base_cr = torch.tensor(float(m['base_cr']), device=device)
    base_p = torch.tensor(float(m['base_price']), device=device)
    cpc_scaling = o.get('cpc_scaling_factor', 0.0)
    effective_cpc = base_cpc * (1 + cpc_scaling * (budget / 50000.0)) 

    max_sessions_t = torch.tensor(m['max_hours'] / m['session_duration'], device=device)
    prices = torch.linspace(o['min_price'], o['max_price'], steps=o['steps'], device=device)

    sim_cr = base_cr * (base_p / prices) ** o['demand_elasticity']
    clients = (budget / effective_cpc) * sim_cr
    demanded_sessions = clients * o['avg_sessions_per_client']
    actual_sessions = torch.clamp(demanded_sessions, max=max_sessions_t)
    revenue = actual_sessions * prices
    taxes = revenue * m.get('tax_rate', 0.06)
    idle_penalty = (max_sessions_t - actual_sessions) * o.get('opportunity_cost_per_hour', 0.0)
    net_profit = revenue - budget - m['fixed_costs'] - taxes
    optimization_target = net_profit - idle_penalty

    best_idx = torch.argmax(optimization_target)
    
    opt_p = prices[best_idx].item()
    final_profit = net_profit[best_idx].item()
    final_vol = actual_sessions[best_idx].item()
    final_tax = taxes[best_idx].item()
    
    final_cpa = (budget / clients[best_idx]).item() if clients[best_idx] > 0 else 0
    final_ltv = opt_p * o['avg_sessions_per_client']

    print(f"\n" + "="*65)
    print(f"   ОТЧЕТ ПО ОПТИМИЗАЦИИ (ВЕРСИЯ 2026.3)")
    print(f"="*65)
    print(f"Рекомендуемая цена:    {opt_p:,.0f} руб.")
    print(f"Ожидаемая прибыль:     {final_profit:,.0f} руб./мес.")
    print(f"Налоги (авторасчет):   {final_tax:,.0f} руб.")
    print(f"Загрузка графика:      {final_vol:,.1f} ч. / {m['max_hours']} ч.")
    print(f"Эффективный CPC:       {effective_cpc.item():,.2f} руб. (с учетом масштаба)")
    print(f"CPA (стоимость лида):  {final_cpa:,.0f} руб.")
    print(f"LTV (ценность лида):   {final_ltv:,.0f} руб.")
    print(f"-"*65)
    print(f"ПЛАН ДЕЙСТВИЙ И РЕШЕНИЯ:")
    
    # !!! Вызов новой функции с расширенными советами !!!
    advices = generate_expert_advice(m, opt_p, final_profit, final_vol, max_sessions_t.item(), final_cpa, final_ltv)
    for i, msg in enumerate(advices, 1):
        print(f" {i}. {msg}")
    print(f"="*65)

if __name__ == "__main__":
    # Убедитесь, что функция generate_expert_advice доступна здесь
    run_optimization()

