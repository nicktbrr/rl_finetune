import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
from sklearn.decomposition import PCA
import pandas as pd

def create_metrics_plot(history: List[Dict]) -> go.Figure:
    """Create a plot showing accuracy, false positives, and false negatives over time."""
    metrics = {
        'accuracy': [],
        'false_positives': [],
        'false_negatives': [],
        'reward': []
    }
    
    for i in range(1, len(history) + 1):
        subset = history[:i]
        correct = sum(1 for h in subset if h['action'] == h['true_label'])
        fp = sum(1 for h in subset if h['action'] == 1 and h['true_label'] == 0)
        fn = sum(1 for h in subset if h['action'] == 0 and h['true_label'] == 1)
        reward = sum(h['reward'] for h in subset)
        
        metrics['accuracy'].append(correct / i)
        metrics['false_positives'].append(fp)
        metrics['false_negatives'].append(fn)
        metrics['reward'].append(reward)
    
    fig = go.Figure()
    
    # Add accuracy line
    fig.add_trace(go.Scatter(
        y=metrics['accuracy'],
        name='Accuracy',
        line=dict(color='green')
    ))
    
    # Add false positives line
    fig.add_trace(go.Scatter(
        y=metrics['false_positives'],
        name='False Positives',
        line=dict(color='red')
    ))
    
    # Add false negatives line
    fig.add_trace(go.Scatter(
        y=metrics['false_negatives'],
        name='False Negatives',
        line=dict(color='orange')
    ))
    
    # Add reward line
    fig.add_trace(go.Scatter(
        y=metrics['reward'],
        name='Total Reward',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Training Metrics Over Time',
        xaxis_title='Number of Points',
        yaxis_title='Count/Score',
        hovermode='x unified'
    )
    
    return fig

def create_decision_boundary_plot(
    data: np.ndarray,
    labels: np.ndarray,
    history: List[Dict],
    n_points: int = 1000
) -> go.Figure:
    """Create a 2D visualization of the decision boundary using PCA."""
    # Reduce dimensionality to 2D
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    # Create meshgrid for decision boundary
    x_min, x_max = data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1
    y_min, y_max = data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add original data points
    fig.add_trace(go.Scatter(
        x=data_2d[:, 0],
        y=data_2d[:, 1],
        mode='markers',
        marker=dict(
            color=labels,
            colorscale=[[0, 'blue'], [1, 'red']],
            size=8
        ),
        name='Data Points'
    ))
    
    # Add points that have been classified
    if history:
        classified_indices = [h['index'] for h in history]
        classified_points = data_2d[classified_indices]
        classified_labels = [h['action'] for h in history]
        
        fig.add_trace(go.Scatter(
            x=classified_points[:, 0],
            y=classified_points[:, 1],
            mode='markers',
            marker=dict(
                color=classified_labels,
                colorscale=[[0, 'lightblue'], [1, 'pink']],
                size=12,
                symbol='circle-open'
            ),
            name='Classified Points'
        ))
    
    fig.update_layout(
        title='2D PCA Projection of Data Points',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        showlegend=True
    )
    
    return fig

def create_reward_plot(history: List[Dict]) -> go.Figure:
    """Create a plot showing individual rewards over time."""
    rewards = [h['reward'] for h in history]
    cumulative_rewards = np.cumsum(rewards)
    
    fig = go.Figure()
    
    # Add individual rewards
    fig.add_trace(go.Scatter(
        y=rewards,
        name='Individual Rewards',
        line=dict(color='blue')
    ))
    
    # Add cumulative rewards
    fig.add_trace(go.Scatter(
        y=cumulative_rewards,
        name='Cumulative Rewards',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title='Rewards Over Time',
        xaxis_title='Number of Points',
        yaxis_title='Reward',
        hovermode='x unified'
    )
    
    return fig

def plot_training_progress(history: list) -> go.Figure:
    """
    Create a plot showing training progress over time.
    
    Args:
        history: List of dictionaries containing training metrics
        
    Returns:
        Plotly figure object
    """
    # Calculate cumulative metrics
    metrics = {
        'accuracy': [],
        'false_positive_rate': [],
        'false_negative_rate': [],
        'total_reward': []
    }
    
    for i in range(1, len(history) + 1):
        subset = history[:i]
        last_entry = subset[-1]
        metrics['accuracy'].append(last_entry['accuracy'])
        metrics['false_positive_rate'].append(last_entry['false_positive_rate'])
        metrics['false_negative_rate'].append(last_entry['false_negative_rate'])
        metrics['total_reward'].append(last_entry['total_reward'])
    
    fig = go.Figure()
    
    # Add traces for each metric
    for metric_name, values in metrics.items():
        fig.add_trace(go.Scatter(
            y=values,
            name=metric_name.replace('_', ' ').title(),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Training Progress',
        xaxis_title='Steps',
        yaxis_title='Value',
        showlegend=True
    )
    
    return fig

def plot_decision_boundary(data: np.ndarray, labels: np.ndarray, 
                         predictions: np.ndarray = None) -> go.Figure:
    """
    Create a 2D visualization of the decision boundary using PCA.
    
    Args:
        data: Input features
        labels: True labels
        predictions: Optional model predictions
        
    Returns:
        Plotly figure object
    """
    # Reduce dimensionality to 2D
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    fig = go.Figure()
    
    # Plot data points
    fig.add_trace(go.Scatter(
        x=data_2d[:, 0],
        y=data_2d[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=labels,
            colorscale='RdYlBu',
            showscale=True,
            colorbar=dict(title='Label')
        ),
        name='Data Points'
    ))
    
    # Add predictions if available
    if predictions is not None:
        fig.add_trace(go.Scatter(
            x=data_2d[:, 0],
            y=data_2d[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=predictions,
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title='Prediction')
            ),
            name='Predictions'
        ))
    
    fig.update_layout(
        title='Decision Boundary Visualization (PCA)',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        showlegend=True
    )
    
    return fig 