import streamlit as st
import plotly.graph_objects as go

def handle_plot_click(fig):
    """Handle click events on the Plotly chart."""
    if fig:
        # Get the clicked point data
        clicked_data = fig.clickData
        if clicked_data and isinstance(clicked_data, list) and len(clicked_data) > 0:
            try:
                # Extract the point index from customdata
                clicked_point = clicked_data[0]
                if isinstance(clicked_point, dict) and 'customdata' in clicked_point:
                    clicked_idx = clicked_point['customdata']
                    # Store the clicked point index in session state
                    st.session_state.clicked_point = clicked_idx
                    # Add to selected points if not already selected
                    if 'selected_points' not in st.session_state:
                        st.session_state.selected_points = []
                    if clicked_idx not in st.session_state.selected_points:
                        st.session_state.selected_points.append(clicked_idx)
                    st.rerun()  # Force a rerun to update the visualization
            except Exception as e:
                st.error(f"Error processing click: {str(e)}")
                st.session_state.clicked_point = None 