import torch

# 1. Настройка окружения
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используем устройство: {device}")

n_scenarios = 1_000_000

# 2. Создаем пространство сценариев (Сингум рынка)
# CAC: нормальное распределение (среднее 200, отклонение 50)
cac = torch.normal(200.0, 50.0, (n_scenarios,), device=device)

# Конверсия: равномерное распределение от 0.01 до 0.05
# В PyTorch используем torch.rand (0..1) и масштабируем его
conversion = 0.01 + torch.rand(n_scenarios, device=device) * (0.05 - 0.01)

# Спектр цен: от 500 до 5000
price = torch.linspace(500, 5000, n_scenarios, device=device)

# 3. Оператор Юнит-Экономики (Холо-функция)
def calculate_unit_stability(p, c_ac, conv):
    # Нелинейная себестоимость (эффект масштаба)
    # Используем torch.exp для мгновенного обсчета всего тензора
    cogs = 1000 * torch.exp(-0.0001 * (conv * 10000)) 
    
    # Чистая прибыль (Маржа)
    # Вычисляется мгновенно для 1 000 000 сценариев параллельно
    contribution_margin = p - cogs - (c_ac / conv)
    return contribution_margin

# 4. Мгновенный расчет
with torch.inference_mode(): # Режим быстрой работы без градиентов
    margins = calculate_unit_stability(price, cac, conversion)

# Находим точку оптимальной "кристаллизации" прибыли
optimal_idx = torch.argmax(margins)
max_profit = margins[optimal_idx]

print("-" * 30)
print(f"Результат 'Квантового скачка' расчетов:")
print(f"Оптимальная цена: {price[optimal_idx]:.2f} руб.")
print(f"Ожидаемая прибыль в этой точке: {max_profit:.2f} руб.")
print(f"Всего просчитано сценариев: {n_scenarios}")
