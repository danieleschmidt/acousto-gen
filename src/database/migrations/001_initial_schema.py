"""
Initial database schema migration for Acousto-Gen.
Creates all tables for optimization results, field data, and experiment tracking.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create initial database schema."""
    
    # Create optimization_results table
    op.create_table(
        'optimization_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_id', sa.String(36), nullable=False),
        sa.Column('method', sa.String(50), nullable=False),
        sa.Column('target_type', sa.String(50), nullable=False),
        sa.Column('iterations', sa.Integer(), nullable=False),
        sa.Column('convergence_threshold', sa.Float(), nullable=True),
        sa.Column('final_loss', sa.Float(), nullable=False),
        sa.Column('converged', sa.Boolean(), nullable=True),
        sa.Column('time_elapsed', sa.Float(), nullable=False),
        sa.Column('phases_json', sa.JSON(), nullable=True),
        sa.Column('phases_binary', sa.LargeBinary(), nullable=True),
        sa.Column('num_elements', sa.Integer(), nullable=False),
        sa.Column('target_specification', sa.JSON(), nullable=False),
        sa.Column('convergence_history', sa.Text(), nullable=True),
        sa.Column('focus_error', sa.Float(), nullable=True),
        sa.Column('peak_pressure', sa.Float(), nullable=True),
        sa.Column('contrast_ratio', sa.Float(), nullable=True),
        sa.Column('efficiency', sa.Float(), nullable=True),
        sa.Column('device_used', sa.String(20), nullable=False),
        sa.Column('memory_usage_mb', sa.Float(), nullable=True),
        sa.Column('experiment_run_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for optimization_results
    op.create_index('idx_optimization_method', 'optimization_results', ['method'])
    op.create_index('idx_optimization_target_type', 'optimization_results', ['target_type'])
    op.create_index('idx_optimization_created_at', 'optimization_results', ['created_at'])
    op.create_index('idx_optimization_final_loss', 'optimization_results', ['final_loss'])
    op.create_index(op.f('ix_optimization_results_run_id'), 'optimization_results', ['run_id'], unique=True)
    
    # Create acoustic_field_data table
    op.create_table(
        'acoustic_field_data',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('field_id', sa.String(36), nullable=False),
        sa.Column('field_type', sa.String(50), nullable=False),
        sa.Column('shape_x', sa.Integer(), nullable=False),
        sa.Column('shape_y', sa.Integer(), nullable=False),
        sa.Column('shape_z', sa.Integer(), nullable=False),
        sa.Column('resolution', sa.Float(), nullable=False),
        sa.Column('frequency', sa.Float(), nullable=False),
        sa.Column('bounds_json', sa.JSON(), nullable=False),
        sa.Column('field_data_path', sa.String(500), nullable=True),
        sa.Column('field_statistics', sa.JSON(), nullable=True),
        sa.Column('max_pressure', sa.Float(), nullable=True),
        sa.Column('mean_pressure', sa.Float(), nullable=True),
        sa.Column('rms_pressure', sa.Float(), nullable=True),
        sa.Column('dynamic_range_db', sa.Float(), nullable=True),
        sa.Column('optimization_result_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for acoustic_field_data
    op.create_index('idx_field_type', 'acoustic_field_data', ['field_type'])
    op.create_index('idx_field_frequency', 'acoustic_field_data', ['frequency'])
    op.create_index('idx_field_created_at', 'acoustic_field_data', ['created_at'])
    op.create_index(op.f('ix_acoustic_field_data_field_id'), 'acoustic_field_data', ['field_id'], unique=True)
    
    # Create array_configurations table
    op.create_table(
        'array_configurations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('config_id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('array_type', sa.String(50), nullable=False),
        sa.Column('num_elements', sa.Integer(), nullable=False),
        sa.Column('frequency', sa.Float(), nullable=False),
        sa.Column('positions_json', sa.JSON(), nullable=False),
        sa.Column('orientations_json', sa.JSON(), nullable=True),
        sa.Column('phase_offsets', sa.JSON(), nullable=True),
        sa.Column('amplitude_factors', sa.JSON(), nullable=True),
        sa.Column('calibration_date', sa.DateTime(), nullable=True),
        sa.Column('calibration_quality', sa.Float(), nullable=True),
        sa.Column('hardware_id', sa.String(100), nullable=True),
        sa.Column('driver_version', sa.String(20), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for array_configurations
    op.create_index('idx_array_type', 'array_configurations', ['array_type'])
    op.create_index('idx_array_active', 'array_configurations', ['is_active'])
    op.create_index('idx_array_num_elements', 'array_configurations', ['num_elements'])
    op.create_index(op.f('ix_array_configurations_config_id'), 'array_configurations', ['config_id'], unique=True)
    
    # Create experiment_runs table
    op.create_table(
        'experiment_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('experiment_id', sa.String(36), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('experiment_type', sa.String(50), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(20), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('total_runs', sa.Integer(), nullable=True),
        sa.Column('successful_runs', sa.Integer(), nullable=True),
        sa.Column('best_result_id', sa.Integer(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('user_id', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for experiment_runs
    op.create_index('idx_experiment_status', 'experiment_runs', ['status'])
    op.create_index('idx_experiment_type', 'experiment_runs', ['experiment_type'])
    op.create_index('idx_experiment_user', 'experiment_runs', ['user_id'])
    op.create_index('idx_experiment_started', 'experiment_runs', ['started_at'])
    op.create_index(op.f('ix_experiment_runs_experiment_id'), 'experiment_runs', ['experiment_id'], unique=True)
    
    # Create user_sessions table
    op.create_table(
        'user_sessions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(36), nullable=False),
        sa.Column('user_id', sa.String(100), nullable=False),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('last_activity', sa.DateTime(), nullable=True),
        sa.Column('session_data', sa.JSON(), nullable=True),
        sa.Column('api_calls', sa.Integer(), nullable=True),
        sa.Column('optimizations_run', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for user_sessions
    op.create_index('idx_session_user', 'user_sessions', ['user_id'])
    op.create_index('idx_session_active', 'user_sessions', ['is_active'])
    op.create_index('idx_session_activity', 'user_sessions', ['last_activity'])
    op.create_index(op.f('ix_user_sessions_session_id'), 'user_sessions', ['session_id'], unique=True)
    
    # Create system_metrics table
    op.create_table(
        'system_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('cpu_usage', sa.Float(), nullable=True),
        sa.Column('memory_usage_mb', sa.Float(), nullable=True),
        sa.Column('gpu_usage', sa.Float(), nullable=True),
        sa.Column('gpu_memory_mb', sa.Float(), nullable=True),
        sa.Column('active_sessions', sa.Integer(), nullable=True),
        sa.Column('optimizations_per_hour', sa.Float(), nullable=True),
        sa.Column('average_optimization_time', sa.Float(), nullable=True),
        sa.Column('hardware_connected', sa.Boolean(), nullable=True),
        sa.Column('hardware_status', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for system_metrics
    op.create_index('idx_metrics_timestamp', 'system_metrics', ['timestamp'])
    
    # Create foreign key constraints
    op.create_foreign_key(
        None, 'optimization_results', 'experiment_runs',
        ['experiment_run_id'], ['id']
    )
    op.create_foreign_key(
        None, 'acoustic_field_data', 'optimization_results',
        ['optimization_result_id'], ['id']
    )
    op.create_foreign_key(
        None, 'experiment_runs', 'optimization_results',
        ['best_result_id'], ['id']
    )


def downgrade():
    """Drop all tables."""
    
    # Drop foreign key constraints first
    op.drop_constraint(None, 'experiment_runs', type_='foreignkey')
    op.drop_constraint(None, 'acoustic_field_data', type_='foreignkey')
    op.drop_constraint(None, 'optimization_results', type_='foreignkey')
    
    # Drop tables
    op.drop_table('system_metrics')
    op.drop_table('user_sessions')
    op.drop_table('experiment_runs')
    op.drop_table('array_configurations')
    op.drop_table('acoustic_field_data')
    op.drop_table('optimization_results')