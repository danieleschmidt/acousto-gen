"""
Comparative study framework for acoustic holography algorithms.
Implements benchmarking and performance comparison across methods.
"""

import numpy as np
import time
import json
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from ..models.acoustic_field import AcousticField, FieldMetrics, create_focus_target
from ..optimization.hologram_optimizer import GradientOptimizer, GeneticOptimizer, OptimizationResult
from ..models.neural_hologram_generator import NeuralHologramGenerator
from ..physics.transducers.transducer_array import UltraLeap256, CircularArray
from ..physics.propagation.wave_propagator import WavePropagator


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    algorithm: str
    target_type: str
    optimization_time: float
    convergence_iterations: int
    final_loss: float
    field_quality: Dict[str, float]
    computational_cost: float
    memory_usage: float
    success_rate: float
    reproducibility_score: float
    metadata: Dict[str, Any]


@dataclass
class ComparisonMetrics:
    """Comprehensive comparison metrics."""
    accuracy_scores: Dict[str, float]
    speed_benchmarks: Dict[str, float]
    quality_metrics: Dict[str, Dict[str, float]]
    resource_usage: Dict[str, Dict[str, float]]
    robustness_scores: Dict[str, float]
    scalability_metrics: Dict[str, float]


class BenchmarkAlgorithm(ABC):
    """Abstract base class for benchmarkable algorithms."""
    
    @abstractmethod
    def optimize(
        self,
        target_field: AcousticField,
        transducer_array: Any,
        max_iterations: int = 1000
    ) -> OptimizationResult:
        """Optimize phases for target field."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get algorithm name."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        pass


class GradientBenchmark(BenchmarkAlgorithm):
    """Gradient-based optimization benchmark."""
    
    def __init__(self, optimizer_type: str = "adam", learning_rate: float = 0.01):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
    
    def optimize(
        self,
        target_field: AcousticField,
        transducer_array: Any,
        max_iterations: int = 1000
    ) -> OptimizationResult:
        optimizer = GradientOptimizer(
            num_elements=len(transducer_array.elements),
            optimizer_type=self.optimizer_type,
            learning_rate=self.learning_rate
        )
        
        # Create forward model
        propagator = WavePropagator(
            frequency=transducer_array.frequency,
            bounds=target_field.bounds
        )
        
        def forward_model(phases):
            positions = transducer_array.get_positions()
            amplitudes = np.ones(len(positions))
            field_data = propagator.compute_field_from_sources(
                positions, amplitudes, phases.detach().cpu().numpy()
            )
            return torch.tensor(field_data, dtype=torch.complex64)
        
        import torch
        target_tensor = torch.tensor(target_field.data, dtype=torch.complex64)
        
        return optimizer.optimize(
            forward_model=forward_model,
            target_field=target_tensor,
            iterations=max_iterations
        )
    
    def get_name(self) -> str:
        return f"Gradient-{self.optimizer_type}"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.learning_rate
        }


class GeneticBenchmark(BenchmarkAlgorithm):
    """Genetic algorithm benchmark."""
    
    def __init__(self, population_size: int = 100, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
    
    def optimize(
        self,
        target_field: AcousticField,
        transducer_array: Any,
        max_iterations: int = 1000
    ) -> OptimizationResult:
        optimizer = GeneticOptimizer(
            num_elements=len(transducer_array.elements),
            population_size=self.population_size,
            mutation_rate=self.mutation_rate
        )
        
        # Create fitness function
        propagator = WavePropagator(
            frequency=transducer_array.frequency,
            bounds=target_field.bounds
        )
        
        def fitness_function(phases):
            positions = transducer_array.get_positions()
            amplitudes = np.ones(len(positions))
            field_data = propagator.compute_field_from_sources(
                positions, amplitudes, phases
            )
            
            # MSE loss
            error = np.mean(np.abs(field_data - target_field.data)**2)
            return 1.0 / (1.0 + error)  # Convert to fitness (higher is better)
        
        return optimizer.optimize(
            fitness_function=fitness_function,
            generations=max_iterations
        )
    
    def get_name(self) -> str:
        return "Genetic"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate
        }


class NeuralBenchmark(BenchmarkAlgorithm):
    """Neural network benchmark."""
    
    def __init__(self, model_type: str = "vae"):
        self.model_type = model_type
        self.generator = None
    
    def optimize(
        self,
        target_field: AcousticField,
        transducer_array: Any,
        max_iterations: int = 1000
    ) -> OptimizationResult:
        start_time = time.time()
        
        if self.generator is None:
            # Load or create pretrained model
            self.generator = NeuralHologramGenerator.from_pretrained("acousto-gen-base")
        
        # Extract condition from target field
        condition = self._extract_condition(target_field)
        
        # Generate hologram
        generated_field = self.generator.generate_hologram(condition)
        
        # Convert to phases (simplified - would need proper inverse model)
        phases = np.random.uniform(0, 2*np.pi, len(transducer_array.elements))
        
        time_elapsed = time.time() - start_time
        
        return OptimizationResult(
            phases=phases,
            amplitudes=np.ones(len(phases)),
            final_loss=0.1,  # Would calculate actual loss
            iterations=1,  # Neural generation is single-shot
            time_elapsed=time_elapsed,
            convergence_history=[0.1],
            metadata={"model_type": self.model_type}
        )
    
    def _extract_condition(self, target_field: AcousticField) -> Dict[str, Any]:
        """Extract condition from target field."""
        # Find focus points
        maxima = target_field.find_maxima(threshold=0.8)
        
        focal_points = []
        for maximum in maxima[:4]:  # Up to 4 focal points
            focal_points.append({
                'position': maximum.position.tolist(),
                'pressure': maximum.get_amplitude()
            })
        
        return {
            'focal_points': focal_points,
            'frequency': target_field.frequency,
            'total_power': 1.0
        }
    
    def get_name(self) -> str:
        return f"Neural-{self.model_type}"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"model_type": self.model_type}


class ComparativeStudy:
    """
    Framework for conducting comprehensive comparative studies
    of acoustic holography algorithms.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize comparative study framework.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Algorithm registry
        self.algorithms: Dict[str, BenchmarkAlgorithm] = {}
        
        # Benchmark results
        self.results: List[BenchmarkResult] = []
        
        # Test cases
        self.test_cases: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics = {}
        
    def register_algorithm(self, algorithm: BenchmarkAlgorithm) -> None:
        """Register algorithm for benchmarking."""
        name = algorithm.get_name()
        self.algorithms[name] = algorithm
        print(f"Registered algorithm: {name}")
    
    def add_test_case(
        self,
        name: str,
        target_field: AcousticField,
        transducer_array: Any,
        expected_quality: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Add test case for benchmarking.
        
        Args:
            name: Test case name
            target_field: Target acoustic field
            transducer_array: Transducer array configuration
            expected_quality: Expected quality metrics
        """
        self.test_cases.append({
            'name': name,
            'target_field': target_field,
            'transducer_array': transducer_array,
            'expected_quality': expected_quality or {}
        })
        print(f"Added test case: {name}")
    
    def create_standard_test_suite(self) -> None:
        """Create standard test suite for acoustic holography."""
        # Single focus test
        array = UltraLeap256()
        bounds = [(-0.1, 0.1), (-0.1, 0.1), (0.05, 0.2)]
        
        # Test 1: Single focus at center
        single_focus = create_focus_target(
            position=[0, 0, 0.1],
            pressure=3000,
            bounds=bounds
        )
        self.add_test_case("single_focus_center", single_focus, array)
        
        # Test 2: Off-center focus
        off_center = create_focus_target(
            position=[0.03, 0.02, 0.12],
            pressure=2500,
            bounds=bounds
        )
        self.add_test_case("single_focus_off_center", off_center, array)
        
        # Test 3: Twin trap
        from ..models.acoustic_field import create_multi_focus_target
        twin_trap = create_multi_focus_target(
            focal_points=[
                ([0.02, 0, 0.1], 3000),
                ([-0.02, 0, 0.1], 3000)
            ],
            bounds=bounds
        )
        self.add_test_case("twin_trap", twin_trap, array)
        
        # Test 4: Quadruple trap
        quad_trap = create_multi_focus_target(
            focal_points=[
                ([0.02, 0.02, 0.1], 2000),
                ([-0.02, 0.02, 0.1], 2000),
                ([0.02, -0.02, 0.1], 2000),
                ([-0.02, -0.02, 0.1], 2000)
            ],
            bounds=bounds
        )
        self.add_test_case("quadruple_trap", quad_trap, array)
        
        # Test 5: High-resolution focus
        sharp_focus = create_focus_target(
            position=[0, 0, 0.08],
            pressure=4000,
            width=0.002,  # Very narrow focus
            bounds=bounds
        )
        self.add_test_case("sharp_focus", sharp_focus, array)
        
        print(f"Created standard test suite with {len(self.test_cases)} test cases")
    
    def run_benchmark(
        self,
        algorithm_name: str,
        test_case: Dict[str, Any],
        num_runs: int = 5,
        max_iterations: int = 1000
    ) -> List[BenchmarkResult]:
        """
        Run benchmark for specific algorithm and test case.
        
        Args:
            algorithm_name: Name of algorithm to test
            test_case: Test case specification
            num_runs: Number of runs for statistical significance
            max_iterations: Maximum optimization iterations
            
        Returns:
            List of benchmark results
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not registered")
        
        algorithm = self.algorithms[algorithm_name]
        target_field = test_case['target_field']
        transducer_array = test_case['transducer_array']
        
        results = []
        
        print(f"Running {num_runs} benchmarks for {algorithm_name} on {test_case['name']}...")
        
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")
            
            # Measure memory usage
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Run optimization
                start_time = time.time()
                result = algorithm.optimize(target_field, transducer_array, max_iterations)
                optimization_time = time.time() - start_time
                
                # Measure final memory
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = final_memory - initial_memory
                
                # Compute generated field
                propagator = WavePropagator(
                    frequency=transducer_array.frequency,
                    bounds=target_field.bounds
                )
                
                generated_field_data = propagator.compute_field_from_sources(
                    transducer_array.get_positions(),
                    result.amplitudes or np.ones(len(result.phases)),
                    result.phases
                )
                
                generated_field = AcousticField(
                    data=generated_field_data,
                    bounds=target_field.bounds,
                    resolution=target_field.resolution,
                    frequency=target_field.frequency
                )
                
                # Calculate field quality metrics
                field_quality = FieldMetrics.calculate_field_statistics(generated_field)
                
                # Calculate success rate (if field meets quality threshold)
                success_rate = 1.0 if result.final_loss < 0.1 else 0.0  # Threshold
                
                # Calculate reproducibility (consistency across runs)
                reproducibility_score = 1.0  # Would compare with other runs
                
                # Create benchmark result
                benchmark_result = BenchmarkResult(
                    algorithm=algorithm_name,
                    target_type=test_case['name'],
                    optimization_time=optimization_time,
                    convergence_iterations=result.iterations,
                    final_loss=result.final_loss,
                    field_quality=field_quality,
                    computational_cost=optimization_time,  # Could be more sophisticated
                    memory_usage=memory_usage,
                    success_rate=success_rate,
                    reproducibility_score=reproducibility_score,
                    metadata={
                        'algorithm_params': algorithm.get_parameters(),
                        'run_number': run,
                        'max_iterations': max_iterations
                    }
                )
                
                results.append(benchmark_result)
                
            except Exception as e:
                print(f"    Error in run {run + 1}: {e}")
                # Create failed result
                failed_result = BenchmarkResult(
                    algorithm=algorithm_name,
                    target_type=test_case['name'],
                    optimization_time=float('inf'),
                    convergence_iterations=0,
                    final_loss=float('inf'),
                    field_quality={},
                    computational_cost=float('inf'),
                    memory_usage=0,
                    success_rate=0.0,
                    reproducibility_score=0.0,
                    metadata={'error': str(e), 'run_number': run}
                )
                results.append(failed_result)
        
        return results
    
    def run_full_comparison(
        self,
        num_runs: int = 5,
        max_iterations: int = 1000
    ) -> ComparisonMetrics:
        """
        Run full comparison across all algorithms and test cases.
        
        Args:
            num_runs: Number of runs per algorithm/test case
            max_iterations: Maximum optimization iterations
            
        Returns:
            Comprehensive comparison metrics
        """
        print("Starting full comparison study...")
        
        all_results = []
        
        # Run all combinations
        for test_case in self.test_cases:
            for algorithm_name in self.algorithms:
                results = self.run_benchmark(
                    algorithm_name, test_case, num_runs, max_iterations
                )
                all_results.extend(results)
        
        self.results = all_results
        
        # Analyze results
        metrics = self._analyze_results(all_results)
        
        # Save results
        self._save_results(all_results, metrics)
        
        return metrics
    
    def _analyze_results(self, results: List[BenchmarkResult]) -> ComparisonMetrics:
        """Analyze benchmark results and compute comparison metrics."""
        # Group results by algorithm
        by_algorithm = {}
        for result in results:
            if result.algorithm not in by_algorithm:
                by_algorithm[result.algorithm] = []
            by_algorithm[result.algorithm].append(result)
        
        # Calculate aggregate metrics
        accuracy_scores = {}
        speed_benchmarks = {}
        quality_metrics = {}
        resource_usage = {}
        robustness_scores = {}
        
        for algorithm, alg_results in by_algorithm.items():
            # Accuracy (inverse of final loss)
            valid_losses = [r.final_loss for r in alg_results if np.isfinite(r.final_loss)]
            accuracy_scores[algorithm] = 1.0 / (1.0 + np.mean(valid_losses)) if valid_losses else 0.0
            
            # Speed (inverse of optimization time)
            valid_times = [r.optimization_time for r in alg_results if np.isfinite(r.optimization_time)]
            speed_benchmarks[algorithm] = 1.0 / (1.0 + np.mean(valid_times)) if valid_times else 0.0
            
            # Quality metrics (aggregate field quality)
            quality_values = {}
            for result in alg_results:
                for metric, value in result.field_quality.items():
                    if metric not in quality_values:
                        quality_values[metric] = []
                    if np.isfinite(value):
                        quality_values[metric].append(value)
            
            quality_metrics[algorithm] = {
                metric: np.mean(values) for metric, values in quality_values.items()
            }
            
            # Resource usage
            valid_memory = [r.memory_usage for r in alg_results if np.isfinite(r.memory_usage)]
            valid_compute = [r.computational_cost for r in alg_results if np.isfinite(r.computational_cost)]
            
            resource_usage[algorithm] = {
                'memory': np.mean(valid_memory) if valid_memory else 0.0,
                'compute': np.mean(valid_compute) if valid_compute else 0.0
            }
            
            # Robustness (success rate across different test cases)
            success_rates = [r.success_rate for r in alg_results]
            robustness_scores[algorithm] = np.mean(success_rates)
        
        # Scalability (would require different array sizes)
        scalability_metrics = {alg: 1.0 for alg in by_algorithm}  # Placeholder
        
        return ComparisonMetrics(
            accuracy_scores=accuracy_scores,
            speed_benchmarks=speed_benchmarks,
            quality_metrics=quality_metrics,
            resource_usage=resource_usage,
            robustness_scores=robustness_scores,
            scalability_metrics=scalability_metrics
        )
    
    def _save_results(
        self,
        results: List[BenchmarkResult],
        metrics: ComparisonMetrics
    ) -> None:
        """Save results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Save comparison metrics
        metrics_file = self.results_dir / f"comparison_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Save summary CSV
        df = pd.DataFrame([asdict(r) for r in results])
        csv_file = self.results_dir / f"benchmark_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to {self.results_dir}")
    
    def generate_report(self, metrics: ComparisonMetrics) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            metrics: Comparison metrics
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("ACOUSTIC HOLOGRAPHY ALGORITHM COMPARISON REPORT")
        report.append("=" * 80)
        report.append()
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        report.append(f"Algorithms tested: {len(metrics.accuracy_scores)}")
        report.append(f"Test cases: {len(self.test_cases)}")
        report.append(f"Total benchmark runs: {len(self.results)}")
        report.append()
        
        # Accuracy ranking
        report.append("ACCURACY RANKING")
        report.append("-" * 40)
        accuracy_sorted = sorted(
            metrics.accuracy_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (alg, score) in enumerate(accuracy_sorted, 1):
            report.append(f"{i}. {alg}: {score:.4f}")
        report.append()
        
        # Speed ranking
        report.append("SPEED RANKING")
        report.append("-" * 40)
        speed_sorted = sorted(
            metrics.speed_benchmarks.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (alg, score) in enumerate(speed_sorted, 1):
            report.append(f"{i}. {alg}: {score:.4f}")
        report.append()
        
        # Robustness ranking
        report.append("ROBUSTNESS RANKING")
        report.append("-" * 40)
        robustness_sorted = sorted(
            metrics.robustness_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (alg, score) in enumerate(robustness_sorted, 1):
            report.append(f"{i}. {alg}: {score:.4f}")
        report.append()
        
        # Resource usage
        report.append("RESOURCE USAGE")
        report.append("-" * 40)
        for alg, usage in metrics.resource_usage.items():
            report.append(f"{alg}:")
            report.append(f"  Memory: {usage['memory']:.2f} MB")
            report.append(f"  Compute: {usage['compute']:.4f} units")
        report.append()
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        best_accuracy = accuracy_sorted[0][0]
        best_speed = speed_sorted[0][0]
        best_robustness = robustness_sorted[0][0]
        
        report.append(f"Best overall accuracy: {best_accuracy}")
        report.append(f"Fastest algorithm: {best_speed}")
        report.append(f"Most robust algorithm: {best_robustness}")
        
        if best_accuracy == best_speed == best_robustness:
            report.append(f"WINNER: {best_accuracy} excels in all categories!")
        else:
            report.append("Choose algorithm based on application requirements:")
            report.append(f"  - For highest quality: {best_accuracy}")
            report.append(f"  - For real-time applications: {best_speed}")
            report.append(f"  - For varied conditions: {best_robustness}")
        
        report.append()
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def create_visualizations(self, metrics: ComparisonMetrics) -> None:
        """Create visualization plots for comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        algorithms = list(metrics.accuracy_scores.keys())
        accuracy_values = list(metrics.accuracy_scores.values())
        
        axes[0, 0].bar(algorithms, accuracy_values)
        axes[0, 0].set_title('Algorithm Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Speed comparison
        speed_values = list(metrics.speed_benchmarks.values())
        
        axes[0, 1].bar(algorithms, speed_values)
        axes[0, 1].set_title('Algorithm Speed Comparison')
        axes[0, 1].set_ylabel('Speed Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Resource usage
        memory_values = [metrics.resource_usage[alg]['memory'] for alg in algorithms]
        compute_values = [metrics.resource_usage[alg]['compute'] for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, memory_values, width, label='Memory')
        axes[1, 0].bar(x + width/2, compute_values, width, label='Compute')
        axes[1, 0].set_title('Resource Usage Comparison')
        axes[1, 0].set_ylabel('Usage')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(algorithms, rotation=45)
        axes[1, 0].legend()
        
        # Robustness vs Speed scatter
        robustness_values = list(metrics.robustness_scores.values())
        
        axes[1, 1].scatter(speed_values, robustness_values)
        for i, alg in enumerate(algorithms):
            axes[1, 1].annotate(alg, (speed_values[i], robustness_values[i]))
        axes[1, 1].set_xlabel('Speed Score')
        axes[1, 1].set_ylabel('Robustness Score')
        axes[1, 1].set_title('Speed vs Robustness')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"comparison_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {plot_file}")


def run_acoustic_holography_study():
    """Run a comprehensive acoustic holography algorithm comparison study."""
    print("Initializing Acoustic Holography Comparative Study...")
    
    # Create study instance
    study = ComparativeStudy("benchmark_results")
    
    # Register algorithms
    study.register_algorithm(GradientBenchmark("adam", 0.01))
    study.register_algorithm(GradientBenchmark("sgd", 0.1))
    study.register_algorithm(GradientBenchmark("lbfgs", 0.01))
    study.register_algorithm(GeneticBenchmark(50, 0.1))
    study.register_algorithm(GeneticBenchmark(100, 0.05))
    # study.register_algorithm(NeuralBenchmark("vae"))  # Uncomment when model available
    
    # Create standard test suite
    study.create_standard_test_suite()
    
    # Run comparison
    metrics = study.run_full_comparison(num_runs=3, max_iterations=500)
    
    # Generate report
    report = study.generate_report(metrics)
    print(report)
    
    # Save report
    report_file = study.results_dir / "comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Create visualizations
    study.create_visualizations(metrics)
    
    print(f"\nStudy complete! Results saved to {study.results_dir}")
    
    return study, metrics


if __name__ == "__main__":
    run_acoustic_holography_study()