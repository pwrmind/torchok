import torch
import torch.nn as nn
import yaml
import os
import time
import warnings
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
import concurrent.futures
from tqdm import tqdm
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard не установлен. Мониторинг отключен.")

# ============================================================================
# 1. КЛАССЫ ДЛЯ ХРАНЕНИЯ ДАННЫХ И ВАЛИДАЦИИ
# ============================================================================

@dataclass
class Metrics:
    """Метрики текущего состояния бизнеса"""
    avg_cpc: float
    ad_budget: float
    base_cr: float
    base_price: float
    session_duration: float
    max_hours: float
    fixed_costs: float
    tax_rate: float
    
    def validate(self) -> None:
        """Валидация входных параметров"""
        assert self.avg_cpc > 0, "CPC должен быть положительным"
        assert self.ad_budget > 0, "Бюджет должен быть положительным"
        assert 0 < self.base_cr <= 1, "Конверсия должна быть в диапазоне (0, 1]"
        assert self.base_price > 0, "Базовая цена должна быть положительной"
        assert self.max_hours > 0, "Макс. часы должны быть положительными"
        assert 0 <= self.tax_rate < 1, "Налог должен быть в диапазоне [0, 1)"

@dataclass
class OptimizationParams:
    """Параметры оптимизации"""
    min_price: float
    max_price: float
    demand_elasticity: float
    cpc_scaling_factor: float
    avg_sessions_per_client: float
    opportunity_cost_per_hour: float
    # Добавляем неиспользуемый параметр для обратной совместимости
    steps: int = 1000000  # Параметр для обратной совместимости, не используется
    
    def validate(self) -> None:
        """Валидация параметров оптимизации"""
        assert self.min_price > 0, "Минимальная цена должна быть положительной"
        assert self.max_price > self.min_price, "Макс. цена должна быть больше минимальной"
        assert self.demand_elasticity > 0, "Эластичность должна быть положительной"
        assert 0 <= self.cpc_scaling_factor <= 1, "Коэффициент масштабирования CPC должен быть в [0, 1]"
        assert self.avg_sessions_per_client > 0, "Среднее количество сессий должно быть положительным"
        assert self.steps > 0, "Количество шагов должно быть положительным"

# ============================================================================
# 2. ОСНОВНОЙ КЛАСС МОДЕЛИ С ПОДДЕРЖКОЙ AUTOGRAD
# ============================================================================

class UnitEconomicsModel(nn.Module):
    """Векторизованная модель юнит-экономики с поддержкой CUDA и autograd"""
    
    def __init__(self, metrics: Metrics, params: OptimizationParams):
        super().__init__()
        self.metrics = metrics
        self.params = params
        
        # Регистрируем параметры как буферы для переноса на GPU
        self.register_buffer('avg_cpc', torch.tensor(metrics.avg_cpc))
        self.register_buffer('ad_budget', torch.tensor(metrics.ad_budget))
        self.register_buffer('base_cr', torch.tensor(metrics.base_cr))
        self.register_buffer('base_price', torch.tensor(metrics.base_price))
        self.register_buffer('max_hours', torch.tensor(metrics.max_hours))
        self.register_buffer('fixed_costs', torch.tensor(metrics.fixed_costs))
        self.register_buffer('tax_rate', torch.tensor(metrics.tax_rate))
        self.register_buffer('demand_elasticity', torch.tensor(params.demand_elasticity))
        self.register_buffer('cpc_scaling_factor', torch.tensor(params.cpc_scaling_factor))
        self.register_buffer('avg_sessions_per_client', torch.tensor(params.avg_sessions_per_client))
        self.register_buffer('opportunity_cost_per_hour', torch.tensor(params.opportunity_cost_per_hour))
    
    @torch.jit.export
    def forward(self, prices: torch.Tensor, budgets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычисление метрик для сетки цен и бюджетов
        
        Args:
            prices: Тензор цен [M, N]
            budgets: Тензор бюджетов [M, N]
            
        Returns:
            net_profit: Чистая прибыль [M, N]
            target_score: Целевая функция [M, N]
            efficiency: Загрузка мощностей [M, N]
        """
        # 1. Динамический CPC (с эффектом масштаба)
        effective_cpc = self.avg_cpc * (1.0 + self.cpc_scaling_factor * (budgets / self.ad_budget))
        
        # 2. Эластичность спроса (конверсия зависит от цены)
        # Используем устойчивую к нулю формулу
        price_ratio = torch.where(prices > 0, self.base_price / prices, torch.ones_like(prices))
        sim_cr = self.base_cr * torch.pow(price_ratio, self.demand_elasticity)
        
        # 3. Воронка продаж (векторизованная)
        potential_clients = budgets / effective_cpc
        potential_sessions = potential_clients * sim_cr * self.avg_sessions_per_client
        
        # 4. Ограничение физической емкости
        actual_sessions = torch.clamp(potential_sessions, max=self.max_hours)
        
        # 5. Финансовые показатели
        revenue = actual_sessions * prices
        taxes = revenue * self.tax_rate
        net_profit = revenue - budgets - self.fixed_costs - taxes
        
        # 6. Штраф за простой (opportunity cost)
        idle_penalty = (self.max_hours - actual_sessions) * self.opportunity_cost_per_hour
        
        # 7. Целевая функция и эффективность
        target_score = net_profit - idle_penalty
        efficiency = actual_sessions / self.max_hours
        
        return net_profit, target_score, efficiency
    
    @torch.jit.export
    def compute_at_point(self, price: torch.Tensor, budget: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Вычисление всех метрик в одной точке (для анализа чувствительности)"""
        # Преобразуем скаляры в тензоры с правильной размерностью
        if price.dim() == 0:
            price = price.unsqueeze(0).unsqueeze(0)
        if budget.dim() == 0:
            budget = budget.unsqueeze(0).unsqueeze(0)
        
        net_profit, target_score, efficiency = self.forward(price, budget)
        
        # Дополнительные метрики
        effective_cpc = self.avg_cpc * (1.0 + self.cpc_scaling_factor * (budget / self.ad_budget))
        sim_cr = self.base_cr * torch.pow(self.base_price / price, self.demand_elasticity)
        potential_clients = budget / effective_cpc
        potential_sessions = potential_clients * sim_cr * self.avg_sessions_per_client
        actual_sessions = torch.clamp(potential_sessions, max=self.max_hours)
        revenue = actual_sessions * price
        
        # Избегаем деления на ноль
        with torch.no_grad():
            ltv = price * self.avg_sessions_per_client
            safe_potential_sessions = torch.where(potential_sessions > 0, potential_sessions, torch.tensor(1.0, device=potential_sessions.device))
            cpa = budget / (safe_potential_sessions / self.avg_sessions_per_client)
            margin_share = torch.where(ltv > 0, cpa / ltv, torch.tensor(float('inf'), device=ltv.device))
        
        return {
            'net_profit': net_profit.squeeze(),
            'target_score': target_score.squeeze(),
            'efficiency': efficiency.squeeze(),
            'effective_cpc': effective_cpc.squeeze(),
            'sim_cr': sim_cr.squeeze(),
            'potential_sessions': potential_sessions.squeeze(),
            'actual_sessions': actual_sessions.squeeze(),
            'revenue': revenue.squeeze(),
            'ltv': ltv.squeeze(),
            'cpa': cpa.squeeze(),
            'margin_share': margin_share.squeeze()
        }

# ============================================================================
# 3. ОПТИМИЗАТОР С АДАПТИВНЫМ ПОИСКОМ И КЭШИРОВАНИЕМ
# ============================================================================

class CudaOptimizer:
    """Оптимизатор с адаптивным поиском на CUDA"""
    
    def __init__(self, device: Optional[torch.device] = None, enable_logging: bool = True):
        self.device = device or self._select_best_gpu()
        self.cache = {}
        self.history = []
        self.enable_logging = enable_logging
        
        # Инициализация мониторинга
        self.writer = None
        if TENSORBOARD_AVAILABLE and enable_logging:
            try:
                log_dir = f"./logs/optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(os.path.dirname(log_dir), exist_ok=True)
                self.writer = SummaryWriter(log_dir)
            except Exception as e:
                warnings.warn(f"Не удалось инициализировать TensorBoard: {e}")
    
    def _select_best_gpu(self) -> torch.device:
        """Автоматический выбор лучшего GPU"""
        if not torch.cuda.is_available():
            warnings.warn("CUDA недоступен. Используется CPU.")
            return torch.device("cpu")
        
        # Выбор GPU с максимальной свободной памятью
        gpu_id = 0
        max_memory = 0
        
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            free_memory = props.total_memory - allocated
            if free_memory > max_memory:
                max_memory = free_memory
                gpu_id = i
        
        device_name = torch.cuda.get_device_name(gpu_id)
        free_gb = max_memory / (1024**3)
        print(f"✅ Выбран GPU {gpu_id} ({device_name}), свободно {free_gb:.1f} GB")
        return torch.device(f"cuda:{gpu_id}")
    
    def _create_optimization_grid(self, metrics: Metrics, params: OptimizationParams, 
                                 resolution: Tuple[int, int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Создание сетки для оптимизации"""
        if resolution is None:
            # Адаптивное разрешение на основе параметра steps
            total_steps = min(params.steps, 1000000)  # Ограничиваем для безопасности
            # Эвристика: квадратный корень от общего числа шагов для каждой оси
            grid_size = int((total_steps ** 0.5) // 2)
            price_steps = budget_steps = max(100, grid_size)  # Минимум 100 шагов
        else:
            price_steps, budget_steps = resolution
        
        prices = torch.linspace(
            params.min_price, 
            params.max_price, 
            price_steps, 
            device=self.device
        )
        
        budgets = torch.linspace(
            metrics.ad_budget * 0.3,
            metrics.ad_budget * 3.0,
            budget_steps,
            device=self.device
        )
        
        P, B = torch.meshgrid(prices, budgets, indexing='ij')
        return P, B
    
    @torch.inference_mode()
    def coarse_search(self, model: UnitEconomicsModel, 
                     resolution: Tuple[int, int] = None) -> Tuple[float, float, float]:
        """Грубый поиск на разреженной сетке"""
        with torch.autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', 
                          dtype=torch.float16, enabled=self.device.type == 'cuda'):
            P, B = self._create_optimization_grid(model.metrics, model.params, resolution)
            _, target_score, _ = model(P, B)
            
            best_idx = torch.argmax(target_score)
            price_steps = P.shape[0]
            budget_steps = P.shape[1]
            
            pi = best_idx // budget_steps
            bi = best_idx % budget_steps
            
            opt_p = P[pi, bi].item()
            opt_b = B[pi, bi].item()
            best_score = target_score[pi, bi].item()
        
        return opt_p, opt_b, best_score
    
    def refine_search(self, model: UnitEconomicsModel, initial_p: float, initial_b: float,
                     learning_rate: float = 100.0, iterations: int = 50) -> Tuple[float, float]:
        """Уточнение оптимума с помощью градиентного спуска"""
        # Включаем градиенты для оптимизации
        p = torch.tensor([initial_p], device=self.device, requires_grad=True)
        b = torch.tensor([initial_b], device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([p, b], lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_p, best_b = initial_p, initial_b
        best_score = -float('inf')
        
        progress_bar = None
        if self.enable_logging:
            progress_bar = tqdm(range(iterations), desc="Уточнение градиентами", leave=False)
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # Вычисляем метрики в точке
            metrics = model.compute_at_point(p, b)
            loss = -metrics['target_score']  # Минимизируем отрицательную целевую функцию
            
            # Ограничения с мягкими штрафами
            penalty = torch.tensor(0.0, device=self.device)
            
            # Штраф за выход за границы цены
            min_price = torch.tensor(model.params.min_price, device=self.device)
            max_price = torch.tensor(model.params.max_price, device=self.device)
            
            # Используем правильную форму для операций с тензорами
            penalty = penalty + torch.relu(min_price - p[0]) ** 2 * 1000
            penalty = penalty + torch.relu(p[0] - max_price) ** 2 * 1000
            
            # Штраф за выход за границы бюджета
            min_budget = torch.tensor(model.metrics.ad_budget * 0.3, device=self.device)
            max_budget = torch.tensor(model.metrics.ad_budget * 3.0, device=self.device)
            penalty = penalty + torch.relu(min_budget - b[0]) ** 2 * 1000
            penalty = penalty + torch.relu(b[0] - max_budget) ** 2 * 1000
            
            total_loss = loss + penalty
            total_loss.backward()
            
            # Применяем градиенты
            torch.nn.utils.clip_grad_norm_([p, b], 1.0)
            optimizer.step()
            scheduler.step(total_loss)
            
            # Обновляем лучший результат
            current_score = -loss.item()
            if current_score > best_score:
                best_score = current_score
                best_p, best_b = p.item(), b.item()
            
            # Логирование
            if self.writer and i % 5 == 0:
                self.writer.add_scalar('Refinement/loss', total_loss.item(), i)
                self.writer.add_scalar('Refinement/price', p.item(), i)
                self.writer.add_scalar('Refinement/budget', b.item(), i)
                self.writer.add_scalar('Refinement/score', current_score, i)
            
            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'цена': f'{p.item():.0f}',
                    'бюджет': f'{b.item():.0f}',
                    'оценка': f'{current_score:.0f}'
                })
        
        if progress_bar:
            progress_bar.close()
        
        return best_p, best_b
    
    def sensitivity_analysis(self, model: UnitEconomicsModel, opt_p: float, opt_b: float, 
                            delta: float = 0.01) -> Dict[str, float]:
        """Анализ чувствительности к изменению параметров"""
        try:
            # Включаем градиенты
            p = torch.tensor([opt_p], device=self.device, requires_grad=True)
            b = torch.tensor([opt_b], device=self.device, requires_grad=True)
            
            # Вычисляем градиенты
            metrics = model.compute_at_point(p, b)
            profit = metrics['net_profit']
            profit.backward()
            
            # Градиенты показывают чувствительность
            price_sensitivity = p.grad.item() if p.grad is not None else 0
            budget_sensitivity = b.grad.item() if b.grad is not None else 0
            
            # Эластичность (процентное изменение)
            profit_value = profit.item()
            if abs(profit_value) > 1e-10:
                price_elasticity = price_sensitivity * opt_p / profit_value
                budget_elasticity = budget_sensitivity * opt_b / profit_value
            else:
                price_elasticity = budget_elasticity = 0
            
            return {
                'price_gradient': price_sensitivity,
                'budget_gradient': budget_sensitivity,
                'price_elasticity': price_elasticity,
                'budget_elasticity': budget_elasticity,
                'profit_change_1p_price': price_sensitivity * opt_p * 0.01,
                'profit_change_1p_budget': budget_sensitivity * opt_b * 0.01
            }
        except Exception as e:
            warnings.warn(f"Ошибка анализа чувствительности: {e}")
            return {
                'price_gradient': 0,
                'budget_gradient': 0,
                'price_elasticity': 0,
                'budget_elasticity': 0,
                'profit_change_1p_price': 0,
                'profit_change_1p_budget': 0
            }
    
    def optimize(self, metrics: Metrics, params: OptimizationParams,
                use_adaptive: bool = True, use_mixed_precision: bool = True) -> Dict:
        """Основной метод оптимизации"""
        start_time = time.time()
        
        try:
            # Создаем модель и переносим на устройство
            model = UnitEconomicsModel(metrics, params).to(self.device)
            model.eval()
            
            # Этап 1: Грубый поиск
            if self.enable_logging:
                print("🔍 Этап 1: Грубый поиск на сетке...")
            coarse_start = time.time()
            opt_p, opt_b, best_score = self.coarse_search(model)
            coarse_time = time.time() - coarse_start
            
            if self.enable_logging:
                print(f"   Найдена точка: цена={opt_p:.0f}, бюджет={opt_b:.0f}, оценка={best_score:.0f}")
            
            # Этап 2: Уточнение (если включен адаптивный поиск)
            if use_adaptive:
                if self.enable_logging:
                    print("🎯 Этап 2: Уточнение градиентным спуском...")
                refine_start = time.time()
                opt_p, opt_b = self.refine_search(model, opt_p, opt_b)
                refine_time = time.time() - refine_start
            else:
                refine_time = 0
            
            # Этап 3: Анализ чувствительности
            if self.enable_logging:
                print("📊 Этап 3: Анализ чувствительности...")
            sensitivity = self.sensitivity_analysis(model, opt_p, opt_b)
            
            # Вычисляем финальные метрики
            with torch.no_grad():
                final_metrics = model.compute_at_point(
                    torch.tensor([opt_p], device=self.device),
                    torch.tensor([opt_b], device=self.device)
                )
            
            total_time = (time.time() - start_time) * 1000
            
            # Сохраняем в историю
            result = {
                'optimal_price': opt_p,
                'optimal_budget': opt_b,
                'net_profit': final_metrics['net_profit'].item(),
                'target_score': final_metrics['target_score'].item(),
                'efficiency': final_metrics['efficiency'].item(),
                'ltv': final_metrics['ltv'].item(),
                'cpa': final_metrics['cpa'].item(),
                'margin_share': final_metrics['margin_share'].item(),
                'sensitivity': sensitivity,
                'timing': {
                    'total_ms': total_time,
                    'coarse_search_ms': coarse_time * 1000,
                    'refine_ms': refine_time * 1000 if use_adaptive else 0,
                    'device': str(self.device)
                }
            }
            
            self.history.append(result)
            
            # Логирование в TensorBoard
            if self.writer:
                self.writer.add_scalar('Results/optimal_price', opt_p)
                self.writer.add_scalar('Results/optimal_budget', opt_b)
                self.writer.add_scalar('Results/net_profit', result['net_profit'])
                self.writer.add_scalar('Results/efficiency', result['efficiency'])
                if sensitivity:
                    self.writer.add_scalar('Sensitivity/price_elasticity', sensitivity.get('price_elasticity', 0))
                    self.writer.add_scalar('Sensitivity/budget_elasticity', sensitivity.get('budget_elasticity', 0))
                self.writer.close()
            
            return result
            
        except torch.cuda.OutOfMemoryError:
            print("⚠️ Недостаточно памяти GPU. Уменьшаем разрешение...")
            # Автоматическое падение на CPU с прогресс-баром
            return self._fallback_cpu_optimization(metrics, params)
        
        except Exception as e:
            print(f"❌ Ошибка оптимизации: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _fallback_cpu_optimization(self, metrics: Metrics, params: OptimizationParams) -> Dict:
        """Fallback оптимизация на CPU с прогресс-баром"""
        print("🔄 Запуск оптимизации на CPU...")
        
        # Уменьшаем разрешение для CPU
        price_steps, budget_steps = 200, 200
        prices = torch.linspace(params.min_price, params.max_price, price_steps)
        budgets = torch.linspace(metrics.ad_budget * 0.3, metrics.ad_budget * 3.0, budget_steps)
        
        best_score = -float('inf')
        best_p, best_b = 0, 0
        
        # Построчная обработка с прогресс-баром
        for i in tqdm(range(price_steps), desc="CPU Оптимизация"):
            for j in range(budget_steps):
                p = prices[i].item()
                b = budgets[j].item()
                
                # Упрощенные вычисления (без autograd)
                effective_cpc = metrics.avg_cpc * (1 + params.cpc_scaling_factor * (b / metrics.ad_budget))
                sim_cr = metrics.base_cr * (metrics.base_price / p) ** params.demand_elasticity
                potential_sessions = (b / effective_cpc) * sim_cr * params.avg_sessions_per_client
                actual_sessions = min(potential_sessions, metrics.max_hours)
                
                revenue = actual_sessions * p
                taxes = revenue * metrics.tax_rate
                net_profit = revenue - b - metrics.fixed_costs - taxes
                idle_penalty = (metrics.max_hours - actual_sessions) * params.opportunity_cost_per_hour
                target_score = net_profit - idle_penalty
                
                if target_score > best_score:
                    best_score = target_score
                    best_p, best_b = p, b
        
        efficiency = min(1.0, (best_b / metrics.ad_budget))
        return {
            'optimal_price': best_p,
            'optimal_budget': best_b,
            'net_profit': best_score,
            'target_score': best_score,
            'efficiency': efficiency,
            'ltv': best_p * params.avg_sessions_per_client,
            'cpa': best_b / (best_p * efficiency) if best_p * efficiency > 0 else 0,
            'margin_share': 0.3,  # Примерное значение
            'timing': {'device': 'cpu (fallback)', 'total_ms': 0}
        }

# ============================================================================
# 4. УТИЛИТЫ ДЛЯ ВЫВОДА И ОТЧЕТОВ
# ============================================================================

class ReportGenerator:
    """Генератор отчетов и стратегических рекомендаций"""
    
    @staticmethod
    def print_optimization_report(result: Dict, metrics: Metrics, params: OptimizationParams):
        """Вывод детального отчета"""
        print("\n" + "="*70)
        print("   🚀 CUDA ADAPTIVE OPTIMIZER 2026 (PRODUCTION READY)")
        print("="*70)
        
        # Информация о вычислениях
        timing = result['timing']
        print(f"Вычислительное устройство:  {timing['device'].upper()}")
        
        if 'coarse_search_ms' in timing and timing['coarse_search_ms'] > 0:
            print(f"Грубый поиск:               {timing['coarse_search_ms']:.1f} ms")
        if 'refine_ms' in timing and timing['refine_ms'] > 0:
            print(f"Уточнение градиентами:      {timing['refine_ms']:.1f} ms")
        print(f"Общее время:               {timing['total_ms']:.1f} ms")
        
        print("-"*70)
        print("📊 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
        print(f"  Цена услуги:             {result['optimal_price']:,.0f} руб.")
        print(f"  Рекламный бюджет:        {result['optimal_budget']:,.0f} руб.")
        print(f"  Прогноз чистой прибыли:  {result['net_profit']:,.0f} руб./мес.")
        print(f"  Загрузка мощностей:      {result['efficiency']*100:.1f}%")
        
        # Анализ чувствительности
        if 'sensitivity' in result and result['sensitivity']:
            sens = result['sensitivity']
            print("-"*70)
            print("🎯 АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ:")
            print(f"  Прибыль к цене:          {sens['price_elasticity']:.3f} (1%↑ цены → {sens['profit_change_1p_price']:+.0f} руб.)")
            print(f"  Прибыль к бюджету:       {sens['budget_elasticity']:.3f} (1%↑ бюджета → {sens['profit_change_1p_budget']:+.0f} руб.)")
        
        # Юнит-экономика
        print("-"*70)
        print("💰 ЮНИТ-ЭКОНОМИКА:")
        print(f"  LTV (пожизненная ценность): {result['ltv']:,.0f} руб.")
        print(f"  CPA (стоимость привлечения): {result['cpa']:,.0f} руб.")
        print(f"  Доля маркетинга в LTV:     {result['margin_share']*100:.1f}%")
        
        # Стратегические рекомендации
        print("-"*70)
        print("🎯 СТРАТЕГИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
        
        recommendations = []
        
        # Анализ бюджета
        budget_ratio = result['optimal_budget'] / metrics.ad_budget
        if budget_ratio > 1.2:
            recommendations.append(f" 1. 📈 УВЕЛИЧЬТЕ БЮДЖЕТ до {result['optimal_budget']:,.0f} руб. "
                                  f"(+{budget_ratio-1:.0%}). Рынок позволяет поглотить больше трафика.")
        elif budget_ratio < 0.8:
            recommendations.append(f" 1. 💰 СНИЗЬТЕ БЮДЖЕТ до {result['optimal_budget']:,.0f} руб. "
                                  f"({budget_ratio-1:.0%}). Вы переплачиваете за неэффективный трафик.")
        else:
            recommendations.append(" 1. ✅ БЮДЖЕТ ОПТИМАЛЕН. Поддерживайте текущий уровень инвестиций.")
        
        # Анализ цены
        price_ratio = result['optimal_price'] / metrics.base_price
        if price_ratio > 1.15:
            recommendations.append(f" 2. 🚀 ПОВЫСЬТЕ ЦЕНУ до {result['optimal_price']:,.0f} руб. "
                                  f"(+{price_ratio-1:.0%}). Ценность услуги выше текущей цены.")
        elif price_ratio < 0.85:
            recommendations.append(f" 2. ⚠️  СНИЗЬТЕ ЦЕНУ до {result['optimal_price']:,.0f} руб. "
                                  f"({price_ratio-1:.0%}). Текущая цена выше рыночной.")
        else:
            recommendations.append(" 2. ✅ ЦЕНА ОПТИМАЛЬНА. Сохраняйте текущее ценовое позиционирование.")
        
        # Анализ рентабельности
        margin_share = result['margin_share']
        if margin_share > 0.4:
            recommendations.append(f" 3. 🔴 ВЫСОКИЙ РИСК: CPA составляет {margin_share*100:.1f}% от LTV. "
                                  f"Сфокусируйтесь на повышении конверсии, а не на трафике.")
        elif margin_share > 0.25:
            recommendations.append(f" 3. 🟡 УМЕРЕННЫЙ РИСК: Доля маркетинга {margin_share*100:.1f}% от LTV. "
                                  f"Оптимизируйте воронку продаж.")
        else:
            recommendations.append(f" 3. 🟢 БЕЗОПАСНАЯ ЗОНА: Доля маркетинга всего {margin_share*100:.1f}% от LTV. "
                                  f"Модель устойчива к колебаниям рынка.")
        
        # Вывод рекомендаций
        for rec in recommendations:
            print(rec)
        
        print("="*70 + "\n")

# ============================================================================
# 5. ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def load_config(config_path: str) -> Tuple[Metrics, OptimizationParams]:
    """Загрузка и валидация конфигурации"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Обработка метрик
    metrics_data = cfg['current_metrics']
    metrics = Metrics(**metrics_data)
    
    # Обработка параметров оптимизации (игнорируем лишние поля)
    params_data = cfg['optimization_params']
    
    # Создаем словарь только с нужными полями
    allowed_params = {}
    param_fields = OptimizationParams.__annotations__.keys()
    
    for field in param_fields:
        if field in params_data:
            allowed_params[field] = params_data[field]
        elif field == 'steps':  # Значение по умолчанию
            allowed_params[field] = params_data.get('steps', 1000000)
    
    params = OptimizationParams(**allowed_params)
    
    return metrics, params

def run_advanced_optimization(config_path: str = 'unit_economics.yaml',
                            use_adaptive: bool = True,
                            use_mixed_precision: bool = True,
                            enable_logging: bool = True):
    """Основная функция запуска оптимизации"""
    
    # Проверка файла конфигурации
    if not os.path.exists(config_path):
        print(f"❌ Ошибка: Файл {config_path} не найден")
        print("Создайте файл конфигурации или укажите правильный путь.")
        return
    
    try:
        if enable_logging:
            print(f"📁 Загрузка конфигурации из {config_path}...")
        
        # Загрузка конфигурации
        metrics, params = load_config(config_path)
        
        # Валидация
        metrics.validate()
        params.validate()
        
        if enable_logging:
            print("✅ Конфигурация загружена и валидирована")
            print(f"📊 Параметры: цена {params.min_price:,.0f}-{params.max_price:,.0f}, "
                  f"бюджет {metrics.ad_budget:,.0f}, шаги {params.steps:,}")
        
        # Создание оптимизатора
        optimizer = CudaOptimizer(enable_logging=enable_logging)
        
        # Запуск оптимизации
        if enable_logging:
            print("🚀 Запуск адаптивной оптимизации...")
        
        result = optimizer.optimize(metrics, params, use_adaptive, use_mixed_precision)
        
        # Генерация отчета
        if enable_logging:
            ReportGenerator.print_optimization_report(result, metrics, params)
        
        # Сохранение результатов
        save_results(result, config_path, enable_logging)
        
        return result
        
    except yaml.YAMLError as e:
        print(f"❌ Ошибка загрузки YAML: {e}")
    except AssertionError as e:
        print(f"❌ Ошибка валидации: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()

def save_results(result: Dict, config_path: str, enable_logging: bool = True):
    """Сохранение результатов оптимизации"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"optimization_results_{timestamp}.yaml"
    
    # Подготовка данных для сохранения
    output_data = {
        'timestamp': timestamp,
        'optimization_results': {
            k: (v if not isinstance(v, dict) else {sk: sv for sk, sv in v.items() if sk != 'sensitivity'})
            for k, v in result.items()
        },
        'config_file': config_path
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        if enable_logging:
            print(f"💾 Результаты сохранены в {output_file}")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить результаты: {e}")

def batch_optimize(configs: List[str], max_workers: int = 2, **kwargs):
    """Пакетная оптимизация нескольких сценариев"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(configs))) as executor:
        future_to_config = {
            executor.submit(run_advanced_optimization, config, **kwargs): config 
            for config in configs
        }
        
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                if result:
                    results.append((config, result))
            except Exception as e:
                print(f"❌ Ошибка для конфига {config}: {e}")
    
    return results

# ============================================================================
# 6. ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    # Пример использования различных режимов
    import argparse
    
    parser = argparse.ArgumentParser(description='CUDA Оптимизатор юнит-экономики')
    parser.add_argument('--config', type=str, default='unit_economics.yaml',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Отключить адаптивный поиск (только grid search)')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Отключить mixed precision вычисления')
    parser.add_argument('--batch', type=str, nargs='+',
                       help='Список конфигураций для пакетной обработки')
    parser.add_argument('--quiet', action='store_true',
                       help='Тихий режим (минимальный вывод)')
    parser.add_argument('--workers', type=int, default=2,
                       help='Количество потоков для пакетной обработки')
    
    args = parser.parse_args()
    
    if args.batch:
        print(f"🧮 Запуск пакетной оптимизации {len(args.batch)} конфигураций...")
        batch_optimize(
            args.batch,
            max_workers=args.workers,
            use_adaptive=not args.no_adaptive,
            use_mixed_precision=not args.no_mixed_precision,
            enable_logging=not args.quiet
        )
    else:
        run_advanced_optimization(
            config_path=args.config,
            use_adaptive=not args.no_adaptive,
            use_mixed_precision=not args.no_mixed_precision,
            enable_logging=not args.quiet
        )