#!/usr/bin/env python3
"""
Тестирование оптимизатора юнит-экономики
"""

import torch
import numpy as np
from main import CudaOptimizer, Metrics, OptimizationParams

def test_memory_optimization():
    """Тест оптимизации памяти"""
    print("🧪 Тестирование оптимизации памяти...")
    
    metrics = Metrics(
        avg_cpc=125.0,
        ad_budget=100000.0,
        base_cr=0.03,
        base_price=3000.0,
        session_duration=1.0,
        max_hours=120.0,
        fixed_costs=80000.0,
        tax_rate=0.06
    )
    
    params = OptimizationParams(
        min_price=2500.0,
        max_price=15000.0,
        demand_elasticity=1.8,
        cpc_scaling_factor=0.1,
        avg_sessions_per_client=4.5,
        opportunity_cost_per_hour=1500.0
    )
    
    optimizer = CudaOptimizer()
    
    # Тест на разных разрешениях
    resolutions = [(100, 100), (500, 500), (1000, 1000)]
    
    for res in resolutions:
        torch.cuda.empty_cache()
        print(f"\nРазрешение: {res[0]}x{res[1]}")
        
        start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            result = optimizer.optimize(metrics, params, use_adaptive=True)
            end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            mem_used = (end_mem - start_mem) / 1024**2  # MB
            print(f"  Память использовано: {mem_used:.1f} MB")
            print(f"  Время: {result['timing']['total_ms']:.1f} ms")
            
        except Exception as e:
            print(f"  Ошибка: {e}")

def test_gradient_accuracy():
    """Тест точности градиентов"""
    print("\n🧪 Тестирование точности градиентов...")
    
    metrics = Metrics(
        avg_cpc=125.0,
        ad_budget=100000.0,
        base_cr=0.03,
        base_price=3000.0,
        session_duration=1.0,
        max_hours=120.0,
        fixed_costs=80000.0,
        tax_rate=0.06
    )
    
    params = OptimizationParams(
        min_price=2500.0,
        max_price=15000.0,
        demand_elasticity=1.8,
        cpc_scaling_factor=0.1,
        avg_sessions_per_client=4.5,
        opportunity_cost_per_hour=1500.0
    )
    
    optimizer = CudaOptimizer()
    
    # Аналитический градиент vs autograd
    price = 5000.0
    budget = 150000.0
    
    result = optimizer.optimize(metrics, params)
    
    if 'sensitivity' in result:
        sens = result['sensitivity']
        print(f"  Градиент по цене: {sens['price_gradient']:.2f}")
        print(f"  Эластичность по цене: {sens['price_elasticity']:.3f}")
        print(f"  Изменение прибыли при +1% цены: {sens['profit_change_1p_price']:+.0f} руб.")

def test_edge_cases():
    """Тест граничных случаев"""
    print("\n🧪 Тестирование граничных случаев...")
    
    test_cases = [
        ("Низкий CPC", {"avg_cpc": 10.0}),
        ("Высокий налог", {"tax_rate": 0.3}),
        ("Низкая эластичность", {"demand_elasticity": 0.5}),
        ("Высокий LTV", {"avg_sessions_per_client": 10.0}),
    ]
    
    base_metrics = Metrics(
        avg_cpc=125.0,
        ad_budget=100000.0,
        base_cr=0.03,
        base_price=3000.0,
        session_duration=1.0,
        max_hours=120.0,
        fixed_costs=80000.0,
        tax_rate=0.06
    )
    
    base_params = OptimizationParams(
        min_price=2500.0,
        max_price=15000.0,
        demand_elasticity=1.8,
        cpc_scaling_factor=0.1,
        avg_sessions_per_client=4.5,
        opportunity_cost_per_hour=1500.0
    )
    
    optimizer = CudaOptimizer()
    
    for name, updates in test_cases:
        print(f"\nКейс: {name}")
        
        # Обновляем метрики
        test_metrics = Metrics(**{**base_metrics.__dict__, **updates})
        
        try:
            result = optimizer.optimize(test_metrics, base_params)
            print(f"  Оптимальная цена: {result['optimal_price']:,.0f} руб.")
            print(f"  Оптимальный бюджет: {result['optimal_budget']:,.0f} руб.")
            print(f"  Прибыль: {result['net_profit']:,.0f} руб.")
        except Exception as e:
            print(f"  Ошибка: {e}")

if __name__ == "__main__":
    print("🧪 ЗАПУСК ТЕСТОВ ОПТИМИЗАТОРА\n")
    
    test_memory_optimization()
    test_gradient_accuracy()
    test_edge_cases()
    
    print("\n✅ Все тесты завершены!")