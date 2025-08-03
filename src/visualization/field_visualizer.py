"""
Visualization tools for acoustic fields and hologram patterns.
Provides 2D/3D plotting and real-time visualization capabilities.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class FieldVisualizer:
    """Acoustic field visualization tools."""
    
    def __init__(self, colormap: str = "RdBu_r"):
        """
        Initialize field visualizer.
        
        Args:
            colormap: Default colormap for visualizations
        """
        self.colormap = colormap
        self.figure_counter = 0
    
    def plot_field_slice(
        self,
        field_data: np.ndarray,
        plane: str = "xy",
        position: float = 0,
        bounds: Optional[List[Tuple[float, float]]] = None,
        title: Optional[str] = None,
        show_phase: bool = False,
        figsize: Tuple[int, int] = (12, 5)
    ) -> Figure:
        """
        Plot 2D slice through acoustic field.
        
        Args:
            field_data: Complex field data
            plane: Slice plane ('xy', 'xz', 'yz')
            position: Position along normal axis
            bounds: Physical bounds of the field
            title: Plot title
            show_phase: Whether to show phase plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2 if show_phase else 1, figsize=figsize)
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Get slice based on plane
        if plane == "xy":
            slice_idx = int(position * field_data.shape[2])
            field_slice = field_data[:, :, slice_idx]
            xlabel, ylabel = "X (m)", "Y (m)"
        elif plane == "xz":
            slice_idx = int(position * field_data.shape[1])
            field_slice = field_data[:, slice_idx, :]
            xlabel, ylabel = "X (m)", "Z (m)"
        else:  # yz
            slice_idx = int(position * field_data.shape[0])
            field_slice = field_data[slice_idx, :, :]
            xlabel, ylabel = "Y (m)", "Z (m)"
        
        # Plot amplitude
        amplitude = np.abs(field_slice)
        im1 = axes[0].imshow(
            amplitude.T,
            origin='lower',
            cmap=self.colormap,
            aspect='auto'
        )
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title(f"Pressure Amplitude ({plane} plane)")
        plt.colorbar(im1, ax=axes[0], label="Pressure (Pa)")
        
        # Plot phase if requested
        if show_phase:
            phase = np.angle(field_slice)
            im2 = axes[1].imshow(
                phase.T,
                origin='lower',
                cmap='hsv',
                aspect='auto',
                vmin=-np.pi,
                vmax=np.pi
            )
            axes[1].set_xlabel(xlabel)
            axes[1].set_ylabel(ylabel)
            axes[1].set_title(f"Phase ({plane} plane)")
            plt.colorbar(im2, ax=axes[1], label="Phase (rad)")
        
        if title:
            fig.suptitle(title)
        
        plt.tight_layout()
        return fig
    
    def plot_3d_isosurfaces(
        self,
        field_data: np.ndarray,
        isovalues: Optional[List[float]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        title: Optional[str] = None,
        opacity: float = 0.3
    ) -> go.Figure:
        """
        Create 3D isosurface visualization using Plotly.
        
        Args:
            field_data: 3D field data (amplitude)
            isovalues: List of isosurface values
            bounds: Physical bounds of the field
            title: Plot title
            opacity: Surface opacity
            
        Returns:
            Plotly figure
        """
        if bounds is None:
            bounds = [(-1, 1), (-1, 1), (-1, 1)]
        
        # Create coordinate arrays
        x = np.linspace(bounds[0][0], bounds[0][1], field_data.shape[0])
        y = np.linspace(bounds[1][0], bounds[1][1], field_data.shape[1])
        z = np.linspace(bounds[2][0], bounds[2][1], field_data.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Default isovalues
        if isovalues is None:
            max_val = np.max(np.abs(field_data))
            isovalues = [0.2 * max_val, 0.5 * max_val, 0.8 * max_val]
        
        # Create figure
        fig = go.Figure()
        
        # Add isosurfaces
        for i, isovalue in enumerate(isovalues):
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=np.abs(field_data).flatten(),
                isomin=isovalue,
                isomax=isovalue,
                surface_count=1,
                opacity=opacity,
                colorscale='RdBu',
                showscale=(i == 0),
                name=f"Iso={isovalue:.1f}"
            ))
        
        # Update layout
        fig.update_layout(
            title=title or "3D Acoustic Field",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True
        )
        
        return fig
    
    def plot_transducer_phases(
        self,
        phases: np.ndarray,
        positions: np.ndarray,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Figure:
        """
        Visualize transducer phase pattern.
        
        Args:
            phases: Phase values in radians
            positions: Transducer positions (Nx3)
            title: Plot title
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Reshape phases if grid array
        grid_size = int(np.sqrt(len(phases)))
        if grid_size**2 == len(phases):
            phase_grid = phases.reshape(grid_size, grid_size)
        else:
            # Interpolate to grid for visualization
            from scipy.interpolate import griddata
            xi = np.linspace(positions[:, 0].min(), positions[:, 0].max(), 50)
            yi = np.linspace(positions[:, 1].min(), positions[:, 1].max(), 50)
            Xi, Yi = np.meshgrid(xi, yi)
            phase_grid = griddata(
                positions[:, :2],
                phases,
                (Xi, Yi),
                method='nearest'
            )
        
        # Phase pattern
        im1 = axes[0, 0].imshow(
            phase_grid,
            cmap='hsv',
            vmin=0,
            vmax=2*np.pi,
            origin='lower'
        )
        axes[0, 0].set_title("Phase Pattern")
        axes[0, 0].set_xlabel("Element X")
        axes[0, 0].set_ylabel("Element Y")
        plt.colorbar(im1, ax=axes[0, 0], label="Phase (rad)")
        
        # Phase histogram
        axes[0, 1].hist(phases, bins=50, edgecolor='black')
        axes[0, 1].set_xlabel("Phase (rad)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Phase Distribution")
        axes[0, 1].set_xlim([0, 2*np.pi])
        
        # Unwrapped phase
        unwrapped = np.unwrap(phases.reshape(-1))
        axes[1, 0].plot(unwrapped)
        axes[1, 0].set_xlabel("Element Index")
        axes[1, 0].set_ylabel("Unwrapped Phase (rad)")
        axes[1, 0].set_title("Unwrapped Phase Profile")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Phase gradient
        if grid_size**2 == len(phases):
            grad_y, grad_x = np.gradient(phase_grid)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            im4 = axes[1, 1].imshow(
                gradient_magnitude,
                cmap='viridis',
                origin='lower'
            )
            axes[1, 1].set_title("Phase Gradient Magnitude")
            axes[1, 1].set_xlabel("Element X")
            axes[1, 1].set_ylabel("Element Y")
            plt.colorbar(im4, ax=axes[1, 1], label="Gradient (rad/element)")
        else:
            axes[1, 1].scatter(
                positions[:, 0],
                positions[:, 1],
                c=phases,
                cmap='hsv',
                vmin=0,
                vmax=2*np.pi
            )
            axes[1, 1].set_title("Transducer Positions")
            axes[1, 1].set_xlabel("X (m)")
            axes[1, 1].set_ylabel("Y (m)")
        
        if title:
            fig.suptitle(title)
        
        plt.tight_layout()
        return fig
    
    def animate_field_evolution(
        self,
        field_sequence: List[np.ndarray],
        plane: str = "xy",
        position: float = 0,
        interval: int = 100,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create animated visualization of field evolution.
        
        Args:
            field_sequence: List of field snapshots
            plane: Slice plane for visualization
            position: Position along normal axis
            interval: Animation interval in ms
            save_path: Optional path to save animation
            
        Returns:
            Plotly animated figure
        """
        frames = []
        
        for i, field in enumerate(field_sequence):
            # Extract slice
            if plane == "xy":
                slice_idx = int(position * field.shape[2])
                field_slice = np.abs(field[:, :, slice_idx])
            elif plane == "xz":
                slice_idx = int(position * field.shape[1])
                field_slice = np.abs(field[:, slice_idx, :])
            else:  # yz
                slice_idx = int(position * field.shape[0])
                field_slice = np.abs(field[slice_idx, :, :])
            
            frames.append(go.Frame(
                data=[go.Heatmap(z=field_slice.T, colorscale='RdBu')],
                name=str(i)
            ))
        
        # Create figure with first frame
        fig = go.Figure(
            data=[go.Heatmap(z=np.abs(field_sequence[0][:, :, 0]).T, colorscale='RdBu')],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': interval}}]},
                    {'label': 'Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}}]}
                ]
            }],
            sliders=[{
                'steps': [
                    {'args': [[f.name], {'frame': {'duration': 0}}],
                     'label': f.name, 'method': 'animate'}
                    for f in frames
                ],
                'active': 0,
                'y': 0,
                'len': 0.9,
                'x': 0.05
            }],
            title="Field Evolution Animation"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_particle_trajectories(
        self,
        trajectories: List[np.ndarray],
        workspace_bounds: Optional[List[Tuple[float, float]]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize particle trajectories in 3D.
        
        Args:
            trajectories: List of particle trajectories (each Nx3)
            workspace_bounds: Workspace boundaries
            title: Plot title
            
        Returns:
            Plotly 3D figure
        """
        fig = go.Figure()
        
        # Plot each trajectory
        colors = px.colors.qualitative.Plotly
        for i, trajectory in enumerate(trajectories):
            color = colors[i % len(colors)]
            
            # Trajectory line
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=3),
                name=f"Particle {i+1}"
            ))
            
            # Start and end points
            fig.add_trace(go.Scatter3d(
                x=[trajectory[0, 0]],
                y=[trajectory[0, 1]],
                z=[trajectory[0, 2]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle'),
                name=f"Start {i+1}",
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                z=[trajectory[-1, 2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='square'),
                name=f"End {i+1}",
                showlegend=False
            ))
        
        # Add workspace bounds if provided
        if workspace_bounds:
            self._add_workspace_bounds(fig, workspace_bounds)
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode='cube'
            ),
            title=title or "Particle Trajectories",
            showlegend=True
        )
        
        return fig
    
    def _add_workspace_bounds(
        self,
        fig: go.Figure,
        bounds: List[Tuple[float, float]]
    ):
        """Add workspace boundary visualization."""
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        z_min, z_max = bounds[2]
        
        # Define corners of the box
        corners = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ]
        
        # Define edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
        ]
        
        # Plot edges
        for edge in edges:
            points = [corners[edge[0]], corners[edge[1]]]
            points = np.array(points)
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='lines',
                line=dict(color='gray', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    def create_dashboard(
        self,
        field_data: np.ndarray,
        phases: np.ndarray,
        metrics: Dict[str, Any]
    ) -> go.Figure:
        """
        Create comprehensive dashboard visualization.
        
        Args:
            field_data: 3D field data
            phases: Transducer phases
            metrics: System metrics dictionary
            
        Returns:
            Plotly dashboard figure
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "XY Plane", "XZ Plane", "Phase Pattern",
                "Intensity Profile", "Metrics", "Phase Histogram"
            ),
            specs=[
                [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
                [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'histogram'}]
            ]
        )
        
        # Field slices
        amplitude = np.abs(field_data)
        
        # XY plane
        fig.add_trace(
            go.Heatmap(z=amplitude[:, :, amplitude.shape[2]//2].T, colorscale='RdBu'),
            row=1, col=1
        )
        
        # XZ plane
        fig.add_trace(
            go.Heatmap(z=amplitude[:, amplitude.shape[1]//2, :].T, colorscale='RdBu'),
            row=1, col=2
        )
        
        # Phase pattern
        grid_size = int(np.sqrt(len(phases)))
        if grid_size**2 == len(phases):
            phase_grid = phases.reshape(grid_size, grid_size)
            fig.add_trace(
                go.Heatmap(z=phase_grid, colorscale='HSV'),
                row=1, col=3
            )
        
        # Intensity profile along z-axis
        z_profile = np.mean(amplitude[:, :, :], axis=(0, 1))
        fig.add_trace(
            go.Scatter(y=z_profile, mode='lines'),
            row=2, col=1
        )
        
        # Metrics bar chart
        if metrics:
            metric_names = list(metrics.keys())[:5]  # Top 5 metrics
            metric_values = [metrics[k] for k in metric_names]
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values),
                row=2, col=2
            )
        
        # Phase histogram
        fig.add_trace(
            go.Histogram(x=phases, nbinsx=50),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title="Acoustic Holography Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig