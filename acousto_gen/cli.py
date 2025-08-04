"""Command-line interface for Acousto-Gen."""

import sys
from pathlib import Path
from typing import Optional, List, Tuple
import json
import numpy as np

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

console = Console()
app = typer.Typer(
    name="acousto-gen",
    help="Generative acoustic holography toolkit for creating 3D pressure fields"
)


@app.command()
def version() -> None:
    """Show version information."""
    from acousto_gen import __version__
    console.print(f"🔊 Acousto-Gen version {__version__}", style="bold green")


@app.command()
def demo(
    array_type: str = typer.Option("ultraleap", "--array", "-a", help="Array type (ultraleap, circular, hemispherical)"),
    frequency: float = typer.Option(40000, "--frequency", "-f", help="Frequency in Hz"),
    focus_position: str = typer.Option("0,0,0.1", "--focus", help="Focus position as x,y,z"),
    iterations: int = typer.Option(500, "--iterations", "-i", help="Optimization iterations"),
    visualize: bool = typer.Option(True, "--visualize/--no-visualize", help="Show visualization"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory")
) -> None:
    """Run a complete holography demo with focus generation."""
    
    console.print("🎯 Starting Acousto-Gen Demo", style="bold blue")
    
    try:
        # Parse focus position
        focus_pos = [float(x) for x in focus_position.split(',')]
        if len(focus_pos) != 3:
            raise ValueError("Focus position must be x,y,z")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Initialize array
            task = progress.add_task("Setting up transducer array...", total=None)
            
            if array_type == "ultraleap":
                from physics.transducers.transducer_array import UltraLeap256
                transducer = UltraLeap256()
            elif array_type == "circular":
                from physics.transducers.transducer_array import CircularArray
                transducer = CircularArray(radius=0.1, num_elements=64)
            elif array_type == "hemispherical":
                from physics.transducers.transducer_array import HemisphericalArray
                transducer = HemisphericalArray(radius=0.15, frequency=frequency)
            else:
                raise ValueError(f"Unknown array type: {array_type}")
            
            progress.update(task, description="Initializing hologram optimizer...")
            
            # Create hologram
            from acousto_gen.core import AcousticHologram
            hologram = AcousticHologram(
                transducer=transducer,
                frequency=frequency,
                resolution=2e-3  # 2mm resolution for demo
            )
            
            # Create target field
            progress.update(task, description="Creating target field...")
            target = hologram.create_focus_point(
                position=tuple(focus_pos),
                pressure=3000
            )
            
            # Optimize phases
            progress.update(task, description=f"Optimizing phases ({iterations} iterations)...")
            
            def callback(iteration, loss, phases):
                if iteration % 50 == 0:
                    progress.update(task, description=f"Iteration {iteration}: Loss = {loss:.4f}")
            
            phases = hologram.optimize(
                target=target,
                iterations=iterations,
                learning_rate=0.05,
                callback=callback
            )
            
            progress.update(task, description="Computing final field...")
            final_field = hologram.compute_field()
            
            # Evaluate quality
            progress.update(task, description="Evaluating field quality...")
            metrics = hologram.evaluate_field_quality(
                field=final_field,
                target_position=np.array(focus_pos)
            )
            
            progress.update(task, description="Demo complete!", completed=True)
        
        # Display results
        console.print("\n✨ Optimization Results", style="bold green")
        
        table = Table(title="Field Quality Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        
        metric_units = {
            'focus_error': 'm',
            'peak_pressure': 'Pa',
            'fwhm': 'm',
            'contrast_ratio': '',
            'efficiency': '',
            'sidelobe_ratio': 'dB'
        }
        
        for key, value in metrics.items():
            if key in metric_units:
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}", metric_units[key])
                else:
                    table.add_row(key, str(value), metric_units[key])
        
        console.print(table)
        
        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.mkdir(exist_ok=True)
            
            console.print(f"\n💾 Saving results to {output_path}")
            
            # Save configuration
            hologram.save_configuration(output_path / "hologram_config.json")
            
            # Save field data
            final_field.save(output_path / "acoustic_field.h5")
            
            # Save metrics
            with open(output_path / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            console.print("✅ Results saved successfully")
        
        # Visualization
        if visualize:
            console.print("\n📊 Generating visualizations...")
            
            try:
                from visualization.field_visualizer import FieldVisualizer
                visualizer = FieldVisualizer()
                
                # Field slice
                fig1 = visualizer.plot_field_slice(
                    final_field.data,
                    plane="xy",
                    position=0.5,
                    title="Acoustic Field (XY Plane)"
                )
                
                # Phase pattern
                fig2 = visualizer.plot_transducer_phases(
                    phases,
                    transducer.positions,
                    title="Optimized Phase Pattern"
                )
                
                if output:
                    import matplotlib.pyplot as plt
                    fig1.savefig(output_path / "field_slice.png", dpi=300, bbox_inches='tight')
                    fig2.savefig(output_path / "phase_pattern.png", dpi=300, bbox_inches='tight')
                    console.print("📊 Visualizations saved")
                else:
                    import matplotlib.pyplot as plt
                    plt.show()
                    
            except ImportError:
                console.print("⚠️  Visualization dependencies not available", style="yellow")
        
        console.print("\n🎉 Demo completed successfully!", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def optimize(
    config_file: str = typer.Argument(..., help="Configuration file path"),
    output: str = typer.Option("output", "--output", "-o", help="Output directory"),
    method: str = typer.Option("adam", "--method", "-m", help="Optimization method"),
    iterations: int = typer.Option(1000, "--iterations", "-i", help="Number of iterations")
) -> None:
    """Optimize hologram from configuration file."""
    
    config_path = Path(config_file)
    if not config_path.exists():
        console.print(f"❌ Configuration file not found: {config_file}", style="bold red")
        raise typer.Exit(1)
    
    console.print(f"🔧 Loading configuration from {config_file}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Implementation would load config and run optimization
        console.print("⚠️  Configuration-based optimization not yet implemented", style="yellow")
        console.print("Use 'acousto-gen demo' for a working example")
        
    except Exception as e:
        console.print(f"❌ Error loading configuration: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def arrays() -> None:
    """List available transducer array types."""
    
    console.print("🔊 Available Transducer Arrays", style="bold blue")
    
    table = Table(title="Supported Array Types")
    table.add_column("Type", style="cyan")
    table.add_column("Elements", style="green")
    table.add_column("Description", style="white")
    
    arrays_info = [
        ("ultraleap", "256", "UltraLeap 16x16 grid array (40 kHz)"),
        ("circular", "64", "Circular ring array (configurable)"),
        ("hemispherical", "320", "Hemispherical focused array (1.5 MHz)"),
        ("custom", "Variable", "User-defined geometry")
    ]
    
    for array_type, elements, description in arrays_info:
        table.add_row(array_type, elements, description)
    
    console.print(table)


@app.command()
def benchmark(
    array_type: str = typer.Option("ultraleap", "--array", "-a", help="Array type"),
    iterations: int = typer.Option(100, "--iterations", "-i", help="Optimization iterations"),
    resolution: float = typer.Option(2e-3, "--resolution", "-r", help="Spatial resolution"),
    device: str = typer.Option("cpu", "--device", "-d", help="Computation device (cpu/cuda)")
) -> None:
    """Run performance benchmark."""
    
    console.print("⚡ Running Acousto-Gen Benchmark", style="bold yellow")
    
    try:
        import time
        
        # Setup
        if array_type == "ultraleap":
            from physics.transducers.transducer_array import UltraLeap256
            transducer = UltraLeap256()
        else:
            console.print("⚠️  Only UltraLeap array supported for benchmark", style="yellow")
            return
        
        from acousto_gen.core import AcousticHologram
        
        # Benchmark hologram creation
        start_time = time.time()
        hologram = AcousticHologram(
            transducer=transducer,
            frequency=40e3,
            resolution=resolution,
            device=device
        )
        setup_time = time.time() - start_time
        
        # Benchmark target creation
        start_time = time.time()
        target = hologram.create_focus_point((0, 0, 0.1))
        target_time = time.time() - start_time
        
        # Benchmark optimization
        start_time = time.time()
        phases = hologram.optimize(target, iterations=iterations)
        optim_time = time.time() - start_time
        
        # Benchmark field computation
        start_time = time.time()
        field = hologram.compute_field()
        field_time = time.time() - start_time
        
        # Results
        table = Table(title="Benchmark Results")
        table.add_column("Operation", style="cyan")
        table.add_column("Time (s)", style="green")
        table.add_column("Rate", style="yellow")
        
        table.add_row("Setup", f"{setup_time:.3f}", "")
        table.add_row("Target Creation", f"{target_time:.3f}", "")
        table.add_row("Optimization", f"{optim_time:.3f}", f"{iterations/optim_time:.1f} it/s")
        table.add_row("Field Computation", f"{field_time:.3f}", "")
        
        console.print(table)
        
        # System info
        console.print(f"\n📊 System Info")
        console.print(f"Device: {device}")
        console.print(f"Array: {transducer.name} ({len(transducer.elements)} elements)")
        console.print(f"Resolution: {resolution*1000:.1f} mm")
        console.print(f"Grid size: {field.shape}")
        
    except Exception as e:
        console.print(f"❌ Benchmark failed: {e}", style="bold red")
        raise typer.Exit(1)


def main() -> None:
    """Entry point for CLI."""
    app()