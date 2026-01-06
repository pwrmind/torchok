import torch
import yaml
import os

def run_optimization():
    if not os.path.exists('unit_economics.yaml'):
        print("Ошибка: Создайте файл unit_economics.yaml")
        return

    with open('unit_economics.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        m = cfg['current_metrics']
        o = cfg['optimization_params']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Перенос базовых данных на GPU
    budget = torch.tensor(float(m['ad_budget']), device=device)
    base_cpc = torch.tensor(float(m['avg_cpc']), device=device)
    base_cr = torch.tensor(float(m['base_cr']), device=device)
    base_p = torch.tensor(float(m['base_price']), device=device)
    
    # 1. Корректировка CPC (учитываем перегрев аукциона при больших бюджетах)
    # Формула: чем выше бюджет относительно "базового", тем дороже клик
    cpc_scaling = o.get('cpc_scaling_factor', 0.0)
    effective_cpc = base_cpc * (1 + cpc_scaling * (budget / 50000.0)) 

    # Параметры оптимизации
    max_sessions_t = torch.tensor(m['max_hours'] / m['session_duration'], device=device)
    prices = torch.linspace(o['min_price'], o['max_price'], steps=o['steps'], device=device)

    # --- ВЕКТОРНЫЕ ВЫЧИСЛЕНИЯ ---
    
    # Спрос с учетом эластичности
    sim_cr = base_cr * (base_p / prices) ** o['demand_elasticity']
    
    # Воронка: Клик -> Клиент -> Сессии (LTV)
    clients = (budget / effective_cpc) * sim_cr
    demanded_sessions = clients * o['avg_sessions_per_client']
    
    # Реальный объем (не больше физического лимита)
    actual_sessions = torch.clamp(demanded_sessions, max=max_sessions_t)
    
    # Экономика
    revenue = actual_sessions * prices
    taxes = revenue * m.get('tax_rate', 0.06)
    
    # Штраф за недозагрузку (Opportunity Cost)
    idle_penalty = (max_sessions_t - actual_sessions) * o.get('opportunity_cost_per_hour', 0.0)
    
    # Чистая прибыль для оптимизации (за вычетом всего, включая "цену времени")
    net_profit = revenue - budget - m['fixed_costs'] - taxes
    optimization_target = net_profit - idle_penalty

    # --- РЕЗУЛЬТАТЫ ---
    best_idx = torch.argmax(optimization_target)
    
    opt_p = prices[best_idx].item()
    final_profit = net_profit[best_idx].item()
    final_vol = actual_sessions[best_idx].item()
    final_tax = taxes[best_idx].item()
    
    # Расчет CPA и LTV
    final_cpa = (budget / clients[best_idx]).item() if clients[best_idx] > 0 else 0
    final_ltv = opt_p * o['avg_sessions_per_client']

    print(f"\n" + "="*65)
    print(f"   ОТЧЕТ ПО ОПТИМИЗАЦИИ (ВЕРСИЯ 2026.2)")
    print(f"="*65)
    print(f"Рекомендуемая цена:    {opt_p:,.0f} руб.")
    print(f"Ожидаемая прибыль:     {final_profit:,.0f} руб./мес.")
    print(f"Налоги (авторасчет):   {final_tax:,.0f} руб.")
    print(f"Загрузка графика:      {final_vol:,.1f} ч. / {m['max_hours']} ч.")
    print(f"Эффективный CPC:       {effective_cpc.item():,.2f} руб. (с учетом масштаба)")
    print(f"CPA (стоимость лида):  {final_cpa:,.0f} руб.")
    print(f"LTV (ценность лида):   {final_ltv:,.0f} руб.")
    print(f"-"*65)
    
    # Вывод советов (используя логику из предыдущего примера)
    from __main__ import generate_expert_advice
    advices = generate_expert_advice(m, opt_p, final_profit, final_vol, max_sessions_t.item(), final_cpa, final_ltv)
    for i, msg in enumerate(advices, 1):
        print(f" {i}. {msg}")
    print(f"="*65)

# Вспомогательная функция (перенесена для целостности)
def generate_expert_advice(m, opt_p, profit, vol, max_sessions, cpa, ltv):
    advice = []
    util = vol / max_sessions
    if util > 0.95: advice.append("[!] ПРЕДЕЛ ЕМКОСТИ: Повышайте цену, вы работаете на износ.")
    if cpa > ltv * 0.5: advice.append("[!] МАРКЕТИНГ: Реклама съедает слишком много. Смените креативы.")
    if profit < m['fixed_costs']: advice.append("[X] ОПАСНОСТЬ: Низкая маржа. Бизнес не окупает риски.")
    return advice

if __name__ == "__main__":
    run_optimization()
