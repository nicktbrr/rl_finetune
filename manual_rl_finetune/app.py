import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from data_handler import DataHandler
from manual_env import ManualRLEnvironment
from visualization import plot_training_progress, plot_decision_boundary
from click_handler import handle_plot_click

# Set page config
st.set_page_config(
    page_title="Manual RL Training",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #e0e0e0;
    }
    .feature-value {
        font-family: monospace;
        background-color: #ffffff;
        padding: 2px 8px;
        border-radius: 3px;
        border: 1px solid #e0e0e0;
        color: #000000;
    }
    .feature-name {
        color: #000000;
        font-weight: bold;
    }
    .importance-bar {
        height: 4px;
        background-color: #e0e0e0;
        border-radius: 2px;
        margin-top: 4px;
    }
    .importance-fill {
        height: 100%;
        background-color: #1f77b4;
        border-radius: 2px;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
    }
    .classification-section {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 20px;
        border-top: 1px solid #e0e0e0;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = None
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = None
if 'current_state' not in st.session_state:
    st.session_state.current_state = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'done' not in st.session_state:
    st.session_state.done = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'clicked_point' not in st.session_state:
    st.session_state.clicked_point = None

# Sidebar
with st.sidebar:
    st.title("Settings")
    data_path = st.text_input("Data Path", value="../25_percent")
    
    if st.button("Load Data"):
        try:
            with st.spinner("Loading data..."):
                data_handler = DataHandler()
                data, labels, original_predictions = data_handler.load_data(data_path)
                
                # Create environment
                env = ManualRLEnvironment(data, labels, original_predictions, data_handler.hidden_reps)
                st.session_state.env = env
                st.session_state.data_handler = data_handler
                
                # Reset environment
                state, done = env.reset()
                st.session_state.current_state = state
                st.session_state.done = done
                
                st.success("Data loaded successfully!")
                
                # Show dataset info
                st.subheader("Dataset Information")
                st.write(f"Number of samples: {len(data)}")
                st.write(f"Number of features: {data.shape[1]}")
                st.write(f"Positive samples: {np.sum(labels == 1)}")
                st.write(f"Negative samples: {np.sum(labels == 0)}")
                
                # Show feature names
                if data_handler.feature_names:
                    st.subheader("Feature Names")
                    st.write(", ".join(data_handler.feature_names))
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Main content
if st.session_state.env is None:
    st.title("Manual RL Training")
    st.write("Please load data from the sidebar to begin training.")
else:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Classify Points", "Similar Points", "Training Metrics"])
    
    with tab1:
        st.header("Classify Points")
        
        if not st.session_state.done:
            # Get current point details
            state = st.session_state.current_state
            point_details = st.session_state.data_handler.get_point_details(
                state['current_point'],
                state['current_idx']
            )
            
            # Display point information
            st.subheader("Current Point Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Index:**", state['current_idx'])
                st.write("**Original Prediction:**", f"{state['original_prediction']:.3f}")
                st.write("**True Label:**", "Attack" if state['true_label'] == 1 else "Benign")
            
            with col2:
                st.write("**Hidden Representation:**")
                st.write(f"Shape: {state['hidden_rep'].shape}")
                st.write(f"Mean: {np.mean(state['hidden_rep']):.3f}")
                st.write(f"Std: {np.std(state['hidden_rep']):.3f}")
            
            # Display feature values in a grid
            st.subheader("Most Important Features")
            st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
            
            # Find max importance for normalization
            max_importance = max(detail['importance'] for detail in point_details.values())
            
            for feature, details in point_details.items():
                importance_percent = (details['importance'] / max_importance) * 100
                st.markdown(f"""
                    <div class="metric-card">
                        <div>
                            <span class="feature-name">{feature}</span>
                            <div class="importance-bar">
                                <div class="importance-fill" style="width: {importance_percent}%"></div>
                            </div>
                            <small>Importance: {details['importance']:.3f}</small>
                        </div>
                        <span class="feature-value">{details['value']:.3f}</span>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add a note about feature importance
            st.markdown("""
                <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    Feature importance is calculated based on:
                    <ul>
                        <li>Distance from feature mean (z-score)</li>
                        <li>Feature value magnitude</li>
                        <li>Model prediction confidence</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            # Classification section
            st.markdown('<div class="classification-section">', unsafe_allow_html=True)
            st.subheader("Classification")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Classify as Benign", key="benign"):
                    state, reward, done, info = st.session_state.env.step(0)
                    st.session_state.current_state = state
                    st.session_state.done = done
                    st.session_state.training_history.append(info)
                    st.rerun()
            
            with col2:
                if st.button("Classify as Attack", key="attack"):
                    state, reward, done, info = st.session_state.env.step(1)
                    st.session_state.current_state = state
                    st.session_state.done = done
                    st.session_state.training_history.append(info)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.success("Training completed! Check the metrics tab for results.")
    
    with tab2:
        st.header("Similar Points")
        
        if not st.session_state.done:
            state = st.session_state.current_state
            
            # Use PCA to visualize similar points
            with st.spinner("Computing PCA visualization..."):
                # Get unclassified points
                unclassified_indices = st.session_state.env.get_unclassified_indices()
                
                if unclassified_indices:
                    # Compute PCA only on unclassified points
                    pca = PCA(n_components=2)
                    unclassified_hidden_reps = st.session_state.data_handler.hidden_reps[unclassified_indices]
                    pca_result = pca.fit_transform(unclassified_hidden_reps)
                    
                    # Create scatter plot
                    fig = go.Figure()
                    
                    # Add unclassified points
                    fig.add_trace(go.Scatter(
                        x=pca_result[:, 0],
                        y=pca_result[:, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=st.session_state.env.labels[unclassified_indices],
                            colorscale='RdYlBu',
                            showscale=True
                        ),
                        name='Unclassified Points',
                        customdata=unclassified_indices,  # Store original indices
                        hovertemplate="Point %{customdata}<br>" +
                                    "PC1: %{x:.2f}<br>" +
                                    "PC2: %{y:.2f}<br>" +
                                    "<extra></extra>"
                    ))
                    
                    # Highlight current point
                    current_idx_in_unclassified = unclassified_indices.index(state['current_idx'])
                    fig.add_trace(go.Scatter(
                        x=[pca_result[current_idx_in_unclassified, 0]],
                        y=[pca_result[current_idx_in_unclassified, 1]],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='star'
                        ),
                        name='Current Point'
                    ))
                    
                    fig.update_layout(
                        title='PCA Visualization of Unclassified Points',
                        xaxis_title='First Principal Component',
                        yaxis_title='Second Principal Component',
                        showlegend=True,
                        dragmode='select',  # Enable selection mode
                        selectdirection='any',  # Allow selection in any direction
                        clickmode='event+select'  # Enable both click and selection events
                    )
                    
                    # Initialize selected points in session state if not present
                    if 'selected_points' not in st.session_state:
                        st.session_state.selected_points = []
                    
                    # Display the plot with event handling
                    selected_points = st.session_state.selected_points
                    
                    # Configure the plot for selection
                    fig.update_layout(
                        dragmode='select',
                        hovermode='closest',
                        clickmode='event',
                        selectionrevision=True
                    )
                    
                    # Add selection callback
                    fig.update_traces(
                        mode='markers',
                        unselected=dict(marker=dict(opacity=0.3)),
                        selected=dict(marker=dict(color='red', size=12))
                    )
                    
                    # Display the plot
                    plot_placeholder = st.empty()
                    plot_data = plot_placeholder.plotly_chart(
                        fig,
                        use_container_width=True,
                        key='pca_plot',
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': ['select2d'],
                            'modeBarButtonsToRemove': ['lasso2d']
                        }
                    )
                    
                    # Add selection controls
                    st.subheader("Group Classification")
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"Selected Points: {len(selected_points)}")
                        if selected_points:
                            st.write("Selected point indices:", selected_points)
                    
                    with col2:
                        if st.button("Clear Selection"):
                            st.session_state.selected_points = []
                            st.rerun()
                    
                    with col3:
                        if st.button("Select All"):
                            st.session_state.selected_points = unclassified_indices.copy()
                            st.rerun()
                    
                    # Add a custom event handler for clicks
                    if st.button("Update Selection", key="update_selection"):
                        try:
                            selected_indices = []
                            for trace in fig.data:
                                if hasattr(trace, 'selectedpoints') and trace.selectedpoints:
                                    points_data = trace.customdata
                                    if points_data is not None and len(points_data) > 0:
                                        selected_indices.extend([points_data[i] for i in trace.selectedpoints])
                            
                            if selected_indices:
                                st.session_state.selected_points = selected_indices
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error updating selection: {str(e)}")
                            pass
                    
                    # Add classification buttons for selected points
                    if selected_points:
                        st.markdown('<div class="classification-section">', unsafe_allow_html=True)
                        st.subheader("Classify Selected Points")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Classify Selected as Benign", key="benign_selected"):
                                # Store current state
                                current_state = st.session_state.current_state.copy()
                                current_idx = current_state['current_idx']
                                
                                # Classify all selected points at once
                                state, reward, done, info = st.session_state.env.classify_points(selected_points, 0)
                                st.session_state.current_state = state
                                st.session_state.done = done
                                st.session_state.training_history.append(info)
                                st.session_state.selected_points = []
                                st.rerun()
                        
                        with col2:
                            if st.button("Classify Selected as Attack", key="attack_selected"):
                                # Store current state
                                current_state = st.session_state.current_state.copy()
                                current_idx = current_state['current_idx']
                                
                                # Classify all selected points at once
                                state, reward, done, info = st.session_state.env.classify_points(selected_points, 1)
                                st.session_state.current_state = state
                                st.session_state.done = done
                                st.session_state.training_history.append(info)
                                st.session_state.selected_points = []
                                st.rerun()
                    
                    # Add individual classification buttons
                    st.markdown('<div class="classification-section">', unsafe_allow_html=True)
                    st.subheader("Classify Current Point")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Classify as Benign", key="benign_similar"):
                            state, reward, done, info = st.session_state.env.step(0)
                            st.session_state.current_state = state
                            st.session_state.done = done
                            st.session_state.training_history.append(info)
                            st.rerun()
                    
                    with col2:
                        if st.button("Classify as Attack", key="attack_similar"):
                            state, reward, done, info = st.session_state.env.step(1)
                            st.session_state.current_state = state
                            st.session_state.done = done
                            st.session_state.training_history.append(info)
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.success("All points have been classified!")
        else:
            st.info("Training completed. No more points to show.")
    
    with tab3:
        st.header("Training Metrics")
        
        if st.session_state.training_history:
            # Calculate metrics
            metrics = st.session_state.env.get_metrics()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            
            with col2:
                st.metric("False Positive Rate", f"{metrics['false_positive_rate']:.3f}")
            
            with col3:
                st.metric("False Negative Rate", f"{metrics['false_negative_rate']:.3f}")
            
            with col4:
                st.metric("Total Reward", f"{metrics['total_reward']:.0f}")
            
            # Plot training progress
            history_df = pd.DataFrame(st.session_state.training_history)
            
            # Training progress plot
            fig1 = plot_training_progress(st.session_state.training_history)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Decision boundary plot
            fig2 = plot_decision_boundary(
                st.session_state.env.data,
                st.session_state.env.labels,
                st.session_state.data_handler.original_predictions
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Cumulative reward plot
            fig3 = px.line(history_df, y='total_reward', title='Cumulative Reward')
            st.plotly_chart(fig3, use_container_width=True)
            
            # Compare with baseline
            if st.session_state.data_handler.original_predictions is not None:
                baseline_accuracy = np.mean(
                    (st.session_state.data_handler.original_predictions > 0.5) == 
                    st.session_state.env.labels
                )
                st.write(f"Baseline Model Accuracy: {baseline_accuracy:.3f}")
                st.write(f"RL Model Improvement: {(metrics['accuracy'] - baseline_accuracy):.3f}")
        else:
            st.info("No training history available yet. Start classifying points to see metrics.") 