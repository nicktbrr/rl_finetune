import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

// Color constants
const COLORS = {
  SELECTED: '#ef4444',
  BENIGN: '#22c55e',
  ATTACK: '#dc2626',
  SELECTED_BORDER: '#ffffff',
  UNCERTAIN: '#f59e0b',
  HOVER: '#3b82f6'
};

// Constants for point appearance
const SELECTED_COLOR = 'yellow';
const BENIGN_COLOR = 'green';
const ATTACK_COLOR = 'red';
const UNCERTAIN_COLOR = 'orange';
const HOVER_COLOR = 'blue';
const BORDER_COLOR = 'black';
const SELECTED_SIZE = 12;
const DEFAULT_SIZE = 8;

function App() {
  const [currentState, setCurrentState] = useState(null);
  const [unclassifiedPoints, setUnclassifiedPoints] = useState([]);
  const [selectedPoints, setSelectedPoints] = useState(new Set());
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState('visualization');
  const [currentSinglePoint, setCurrentSinglePoint] = useState(null);
  const [validUnclassifiedPoints, setValidUnclassifiedPoints] = useState([]);
  const [currentPointDetails, setCurrentPointDetails] = useState(null);
  const [hoveredPointDetails, setHoveredPointDetails] = useState(null);
  const [displayedPoint, setDisplayedPoint] = useState(null);
  const [pinnedPoint, setPinnedPoint] = useState(null);

  // Load initial state
  useEffect(() => {
    loadInitialState();
  }, []);

  // Effect to initialize current single point
  useEffect(() => {
    if (validUnclassifiedPoints?.length > 0 && currentSinglePoint === null) {
      handleNextPoint();
    }
  }, [validUnclassifiedPoints, currentSinglePoint]);

  // Process valid unclassified points whenever currentState or unclassifiedPoints change
  useEffect(() => {
    if (currentState?.state?.hidden_reps && Array.isArray(unclassifiedPoints)) {
      // Check if the hidden representations exist and are valid for points
      const isValidHiddenRep = (index) => {
        const hiddenReps = currentState.state.hidden_reps;
        return index < hiddenReps.length && 
               Array.isArray(hiddenReps[index]) && 
               hiddenReps[index].length >= 2;
      };
      
      // Filter out invalid points
      const validPoints = unclassifiedPoints.filter(isValidHiddenRep);
      setValidUnclassifiedPoints(validPoints);
    }
  }, [currentState, unclassifiedPoints]);

  const loadInitialState = async () => {
    setLoading(true);
    try {
      // Get current state
      const stateResponse = await axios.get(`${API_BASE_URL}/current_state`);
      console.log('State response:', stateResponse.data);
      setCurrentState(stateResponse.data);
      
      // Get unclassified points (these are indices directly from the state)
      setUnclassifiedPoints(stateResponse.data.state.unclassified_indices || []);
      console.log('Unclassified points:', stateResponse.data.state.unclassified_indices);
      
      // Get training history
      const historyResponse = await axios.get(`${API_BASE_URL}/training_history`);
      console.log('History response:', historyResponse.data);
      setTrainingHistory(historyResponse.data.history || []);
      
      // Load metrics
      await loadMetrics();
      
      setError(null);
    } catch (err) {
      console.error('Error loading initial state:', err);
      setError('Failed to load initial state. Please try refreshing the page.');
    } finally {
      setLoading(false);
    }
  };

  // Load metrics
  const loadMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/metrics`);
      console.log('Metrics response:', response.data);
      setMetrics(response.data);
    } catch (err) {
      console.error('Error loading metrics:', err);
    }
  };

  const handleClassify = async (label) => {
    if (selectedPoints.size === 0) {
      alert('Please select at least one point to classify');
      return;
    }

    setLoading(true);
    try {
      console.log('Classifying points:', Array.from(selectedPoints), 'with label:', label);
      
      const response = await axios.post(`${API_BASE_URL}/classify_points`, {
        points: Array.from(selectedPoints),
        label: label
      });

      console.log('Classify response:', response.data);
      
      // Update state with new data
      setCurrentState(response.data);
      setUnclassifiedPoints(response.data.state.unclassified_indices || []);
      setSelectedPoints(new Set());
      setTrainingHistory(response.data.history || []);
      
      // Update metrics and history
      await loadMetrics();
      await loadHistory();
      
    } catch (err) {
      console.error('Error classifying points:', err);
      setError('Failed to classify points. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/reset`);
      console.log('Reset response:', response.data);
      
      // Update all state variables with fresh data
      setCurrentState(response.data);
      setSelectedPoints(new Set());
      setHoveredPoint(null);
      
      // Get the unclassified points from the state
      setUnclassifiedPoints(response.data.state.unclassified_indices || []);
      
      // Clear history
      setTrainingHistory([]);
      setHistory([]);
      
      // Update metrics
      await loadMetrics();
      
      setError(null);
    } catch (err) {
      console.error('Error resetting:', err);
      setError('Failed to reset. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Load history
  const loadHistory = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/training_history`);
      console.log('History response:', response.data);
      setHistory(response.data.history || []);
    } catch (err) {
      console.error('Error loading history:', err);
    }
  };

  const handlePlotSelection = (eventData) => {
    console.log('Plot selection event:', eventData);
    
    // If no selection data, do nothing
    if (!eventData || !eventData.points || eventData.points.length === 0) {
      return;
    }
    
    // Create a new set from existing selection
    const newSelected = new Set(selectedPoints);
    
    // For a single click (single point), toggle selection
    if (eventData.points.length === 1) {
      const pointIndex = eventData.points[0].customdata;
      console.log('Toggling single point:', pointIndex);
      
      if (newSelected.has(pointIndex)) {
        newSelected.delete(pointIndex);
      } else {
        newSelected.add(pointIndex);
      }
      
      // Pin or unpin the point for feature display
      if (pinnedPoint === pointIndex) {
        setPinnedPoint(null);
      } else {
        setPinnedPoint(pointIndex);
        setDisplayedPoint(pointIndex);
        
        // Fetch details for the pinned point
        fetchPointDetails(pointIndex).then(details => {
          setHoveredPointDetails(details);
        });
      }
    } 
    // For box/lasso selection, add all points
    else {
      console.log('Box/lasso selection with points:', eventData.points.length);
      eventData.points.forEach(point => {
        const pointIndex = point.customdata;
        newSelected.add(pointIndex);
      });
    }
    
    console.log('New selection set:', Array.from(newSelected));
    setSelectedPoints(newSelected);
  };

  const handlePlotHover = useCallback(async (eventData) => {
    if (!eventData || !eventData.points || eventData.points.length === 0) return;
    
    const point = eventData.points[0];
    if (point.customdata === undefined) return;
    
    // Get the point index directly from customdata
    const pointIndex = point.customdata;
    console.log('Hovering over point:', pointIndex);
    
    setHoveredPoint(pointIndex);
    
    // Only update displayed point if no point is pinned
    if (pinnedPoint === null) {
      setDisplayedPoint(pointIndex);
      
      // Fetch point details when hovering
      try {
        const response = await axios.get(`${API_BASE_URL}/point_details/${pointIndex}`);
        console.log('Hover point details response:', response.data);
        setHoveredPointDetails(response.data.feature_details);
      } catch (err) {
        console.error(`Error fetching hover point details: ${err}`);
      }
    }
  }, [pinnedPoint]);

  const getTopFeatures = useCallback((pointIndex, count = 10) => {
    if (!currentState?.point_details || pointIndex === null) return [];
    
    // Get the point details from the current state
    const details = currentState.point_details;
    
    // Create an array of feature objects from the point details
    const features = [];
    for (const [name, data] of Object.entries(details)) {
      features.push({
        name: name, // Real feature name from backend
        value: data.value,
        importance: data.importance
      });
    }
    
    // Sort by importance (descending)
    features.sort((a, b) => b.importance - a.importance);
    
    // Return the top N features
    return features.slice(0, count);
  }, [currentState]);

  // Handle next point in single point view
  const handleNextPoint = () => {
    // We can safely use unclassifiedPoints since we're no longer using validUnclassifiedPoints directly
    if (!Array.isArray(unclassifiedPoints) || unclassifiedPoints.length === 0) {
      console.log("No unclassified points available");
      return;
    }
    
    // Get a random unclassified point
    const randomIndex = Math.floor(Math.random() * unclassifiedPoints.length);
    const pointIndex = unclassifiedPoints[randomIndex];
    
    // Make sure the point has valid hidden representations
    if (currentState && 
        currentState.state && 
        currentState.state.hidden_reps && 
        pointIndex < currentState.state.hidden_reps.length && 
        Array.isArray(currentState.state.hidden_reps[pointIndex])) {
      console.log(`Setting current single point to ${pointIndex}`);
      setCurrentSinglePoint(pointIndex);
    } else {
      console.log(`Point ${pointIndex} has invalid hidden representations, trying another one`);
      // Try again with another point if there are any left
      if (unclassifiedPoints.length > 1) {
        handleNextPoint();
      }
    }
  };

  // Handle single point classification
  const handleSinglePointClassify = async (label) => {
    if (currentSinglePoint === null) return;

    setLoading(true);
    try {
      console.log('Classifying single point:', currentSinglePoint, 'with label:', label);
      
      const response = await axios.post(`${API_BASE_URL}/classify_points`, {
        points: [currentSinglePoint],
        label: label
      });

      console.log('Classify response:', response.data);
      
      // Update state with new data
      setCurrentState(response.data);
      setUnclassifiedPoints(response.data.state.unclassified_indices || []);
      setTrainingHistory(response.data.history || []);
      
      // Update metrics and history
      await loadMetrics();
      await loadHistory();
      
      // Move to next point
      setCurrentSinglePoint(null);
      handleNextPoint();
      
    } catch (err) {
      console.error('Error classifying point:', err);
      setError('Failed to classify point. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Function to fetch point details for a specific point
  const fetchPointDetails = async (pointIndex) => {
    if (pointIndex === null) return null;
    
    try {
      // Use the endpoint to get point-specific details
      console.log(`Fetching details for point ${pointIndex}`);
      const response = await axios.get(`${API_BASE_URL}/point_details/${pointIndex}`);
      console.log('Point details response:', response.data);
      
      // Return the details
      return response.data.feature_details;
    } catch (err) {
      console.error(`Error fetching details for point ${pointIndex}:`, err);
      return null;
    }
  };

  // Function to fetch point details for a specific point (for single point view)
  const fetchSinglePointDetails = async (pointIndex) => {
    if (pointIndex === null) return;
    
    try {
      // Use the endpoint to get point-specific details
      console.log(`Fetching details for single point ${pointIndex}`);
      const response = await axios.get(`${API_BASE_URL}/point_details/${pointIndex}`);
      console.log('Single point details response:', response.data);
      
      // Set the details for the current point view
      setCurrentPointDetails(response.data.feature_details);
    } catch (err) {
      console.error(`Error fetching details for single point ${pointIndex}:`, err);
      // Fallback to using general feature data
      fallbackPointDetails(pointIndex);
    }
  };

  // Fallback method to extract point features from state
  const fallbackPointDetails = (pointIndex) => {
    if (currentState && currentState.state && currentState.state.features) {
      const pointFeatures = currentState.state.features[pointIndex];
      
      // Create a map of feature names to values and estimated importance
      const details = {};
      if (Array.isArray(pointFeatures)) {
        // Get feature names from the data handler, or use generic names
        const featureNames = currentState.state.feature_names || 
                           (currentState.point_details ? Object.keys(currentState.point_details) : 
                           pointFeatures.map((_, i) => `Feature_${i}`));
        
        // Create feature details with actual values from this specific point
        featureNames.forEach((name, i) => {
          if (i < pointFeatures.length) {
            const value = pointFeatures[i];
            details[name] = {
              value: value,
              importance: Math.abs(value) // Use absolute value as a simple importance measure
            };
          }
        });
      }
      
      setCurrentPointDetails(details);
      console.log("Set fallback point details:", details);
    }
  };

  // Update the useEffect to fetch point details when currentSinglePoint changes
  useEffect(() => {
    if (currentSinglePoint !== null) {
      fetchSinglePointDetails(currentSinglePoint);
    }
  }, [currentSinglePoint]);

  // Update the getTopFeatures function to use point-specific details for the single point view
  const getPointTopFeatures = useCallback((pointIndex, count = 10) => {
    // For single point view, use the point-specific details
    if (pointIndex === currentSinglePoint && currentPointDetails) {
      // Create an array of feature objects from the point-specific details
      const features = [];
      for (const [name, data] of Object.entries(currentPointDetails)) {
        features.push({
          name: name,
          value: data.value,
          importance: data.importance
        });
      }
      
      // Sort by importance (descending)
      features.sort((a, b) => b.importance - a.importance);
      
      // Return the top N features
      return features.slice(0, count);
    }
    
    // Fallback to the general function for other cases
    return getTopFeatures(pointIndex, count);
  }, [currentPointDetails, currentSinglePoint, getTopFeatures]);

  // Add a useEffect to clear hover point details when unhovered
  useEffect(() => {
    if (hoveredPoint === null) {
      setHoveredPointDetails(null);
    }
  }, [hoveredPoint]);

  // Update the getHoverFeatures function to use point-specific details
  const getHoverFeatures = useCallback((pointIndex, count = 5) => {
    // Use the hover point details if available
    if (pointIndex === hoveredPoint && hoveredPointDetails) {
      // Create an array of feature objects from the hover point details
      const features = [];
      for (const [name, data] of Object.entries(hoveredPointDetails)) {
        features.push({
          name: name,
          value: data.value,
          importance: data.importance
        });
      }
      
      // Sort by importance (descending)
      features.sort((a, b) => b.importance - a.importance);
      
      // Return the top N features
      return features.slice(0, count);
    }
    
    // Fallback to the general function for other cases
    return getTopFeatures(pointIndex, count);
  }, [hoveredPointDetails, hoveredPoint, getTopFeatures]);

  // Add function to unpin point
  const unpinPoint = () => {
    setPinnedPoint(null);
  };

  if (error) {
    return (
      <div className="error-container">
        <div className="error-card">
          <svg className="error-icon" viewBox="0 0 24 24">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
          </svg>
          <h2>Error</h2>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (loading || !currentState || !currentState.state || !currentState.state.hidden_reps) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading data...</p>
      </div>
    );
  }

  // Get hiddenReps from currentState
  const hiddenReps = currentState.state.hidden_reps;
  
  // Show an error if no valid points are available
  if (validUnclassifiedPoints.length === 0) {
    return (
      <div className="error-container">
        <div className="error-card">
          <h2>Data Error</h2>
          <p>No valid points to display. The PCA data may be corrupted or missing.</p>
          <button onClick={handleReset} className="retry-button">Reset Application</button>
        </div>
      </div>
    );
  }
  
  const predictions = currentState.state.predictions || [];

  // Helper function to get point color based on prediction
  const getPointColor = (index) => {
    if (hoveredPoint === index) return COLORS.HOVER;
    if (selectedPoints.has(index)) return COLORS.SELECTED;
    
    const prediction = predictions[index];
    if (prediction === undefined) return COLORS.UNCERTAIN;
    if (prediction > 0.7) return COLORS.ATTACK;
    if (prediction < 0.3) return COLORS.BENIGN;
    return COLORS.UNCERTAIN;
  };

  // Helper function to get tooltip text for a point
  const getTooltipText = (index) => {
    const prediction = predictions[index];
    if (prediction === undefined) return 'No prediction available';
    
    const predictionText = `${(prediction * 100).toFixed(1)}% attack probability`;
    return predictionText;
  };

  // Create plot data only with valid points
  const plotData = [{
    x: validUnclassifiedPoints.map(i => hiddenReps[i][0]),
    y: validUnclassifiedPoints.map(i => hiddenReps[i][1]),
    mode: 'markers',
    type: 'scatter',
    marker: {
      size: validUnclassifiedPoints.map(i => 
        selectedPoints.has(i) ? SELECTED_SIZE : DEFAULT_SIZE
      ),
      color: validUnclassifiedPoints.map(i => getPointColor(i)),
      opacity: validUnclassifiedPoints.map(i => 
        selectedPoints.has(i) || hoveredPoint === i ? 1 : 0.7
      ),
      line: {
        color: validUnclassifiedPoints.map(i => 
          selectedPoints.has(i) || hoveredPoint === i 
            ? BORDER_COLOR 
            : 'transparent'
        ),
        width: validUnclassifiedPoints.map(i => 
          selectedPoints.has(i) || hoveredPoint === i ? 2 : 1
        ),
      }
    },
    // Use simple number for customdata to avoid issues
    customdata: validUnclassifiedPoints,
    hovertemplate: '<b>Point %{customdata}</b><br>Attack Probability: %{text}<extra></extra>',
    text: validUnclassifiedPoints.map(i => getTooltipText(i))
  }];

  // Calculate metrics display values
  const metricsDisplay = metrics ? {
    accuracy: (metrics.accuracy * 100).toFixed(2),
    precision: (metrics.precision * 100).toFixed(2),
    recall: (metrics.recall * 100).toFixed(2),
    f1: (metrics.f1 * 100).toFixed(2),
    classified: metrics.total_classified,
    remaining: metrics.remaining_unclassified,
  } : {
    accuracy: '0.00',
    precision: '0.00',
    recall: '0.00',
    f1: '0.00',
    classified: 0,
    remaining: unclassifiedPoints.length,
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Network Traffic Classifier</h1>
        <div className="tabs">
          <button 
            className={`tab-button ${activeTab === 'visualization' ? 'active' : ''}`}
            onClick={() => setActiveTab('visualization')}
          >
            Mass Classification
          </button>
          <button 
            className={`tab-button ${activeTab === 'single-point' ? 'active' : ''}`}
            onClick={() => setActiveTab('single-point')}
          >
            Single Point
          </button>
          <button 
            className={`tab-button ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            Classification History
          </button>
          <button 
            className={`tab-button ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => setActiveTab('metrics')}
          >
            Training Metrics
          </button>
        </div>
      </header>

      {activeTab === 'visualization' && (
        <div className="main-content">
          <div className="metrics-grid">
            <div className="metric-card">
              <h3>Accuracy</h3>
              <div className="metric-value">{metricsDisplay.accuracy}%</div>
            </div>
            <div className="metric-card">
              <h3>Precision</h3>
              <div className="metric-value">{metricsDisplay.precision}%</div>
            </div>
            <div className="metric-card">
              <h3>Recall</h3>
              <div className="metric-value">{metricsDisplay.recall}%</div>
            </div>
            <div className="metric-card">
              <h3>F1 Score</h3>
              <div className="metric-value">{metricsDisplay.f1}%</div>
            </div>
          </div>

          <div className="plots-container">
            <div className="plot-card">
              <Plot
                data={plotData}
                layout={{
                  autosize: true,
                  title: 'PCA Visualization of Network Traffic',
                  xaxis: { 
                    title: 'First Principal Component',
                    gridcolor: '#e5e7eb',
                  },
                  yaxis: { 
                    title: 'Second Principal Component',
                    gridcolor: '#e5e7eb',
                  },
                  paper_bgcolor: 'white',
                  plot_bgcolor: 'white',
                  hovermode: 'closest',
                  dragmode: 'select',
                  showlegend: false,
                  margin: { t: 60, r: 40, b: 60, l: 60 },
                  selectdirection: 'any',
                }}
                config={{
                  responsive: true,
                  displayModeBar: true,
                  modeBarButtonsToRemove: ['lasso2d', 'pan2d', 'autoScale2d'],
                  displaylogo: false,
                }}
                onSelected={handlePlotSelection}
                onHover={handlePlotHover}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
              />
              <div className="plot-controls">
                <button 
                  onClick={() => handleClassify(0)} 
                  disabled={loading || selectedPoints.size === 0}
                  className="classify-button benign"
                >
                  Classify as Benign
                </button>
                <button 
                  onClick={() => handleClassify(1)} 
                  disabled={loading || selectedPoints.size === 0}
                  className="classify-button attack"
                >
                  Classify as Attack
                </button>
                <div className="selection-info">
                  Selected: {selectedPoints.size} points
                </div>
              </div>
            </div>

            <div className="feature-info">
              <h3>
                {displayedPoint !== null ? (
                  <>
                    Feature Analysis for Point {displayedPoint}
                    {pinnedPoint === displayedPoint && (
                      <span className="pin-indicator">
                        (Pinned)
                        <button className="unpin-button" onClick={unpinPoint}>
                          Unpin
                        </button>
                      </span>
                    )}
                  </>
                ) : 'Feature Analysis'}
              </h3>
              {displayedPoint !== null ? (
                <div className="feature-list">
                  {getHoverFeatures(displayedPoint, 5).map((feature, idx) => (
                    <div key={idx} className="feature-item">
                      <div className="feature-name" data-name={feature.name}>{feature.name}</div>
                      <div className="feature-bar-container">
                        <div 
                          className={`feature-bar ${feature.value > 0 ? 'positive' : 'negative'}`}
                          style={{ width: `${Math.min(Math.abs(feature.importance) * 100, 100)}%` }}
                        ></div>
                      </div>
                      <div className="feature-value">
                        {feature.value.toFixed(4)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-features">
                  <p>Hover over or select a point to see feature details</p>
                </div>
              )}
            </div>
          </div>

          <div className="progress-section">
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${(metricsDisplay.classified / (metricsDisplay.classified + metricsDisplay.remaining)) * 100}%` }}
              ></div>
            </div>
            <div className="progress-info">
              {metricsDisplay.classified} of {metricsDisplay.classified + metricsDisplay.remaining} points classified 
              ({((metricsDisplay.classified / (metricsDisplay.classified + metricsDisplay.remaining)) * 100).toFixed(1)}%)
            </div>
          </div>

          <button onClick={handleReset} className="reset-button">
            Reset Environment
          </button>
        </div>
      )}

      {activeTab === 'single-point' && (
        <div className="main-content">
          <div className="metrics-grid">
            <div className="metric-card">
              <h3>Accuracy</h3>
              <div className="metric-value">{metricsDisplay.accuracy}%</div>
            </div>
            <div className="metric-card">
              <h3>Classified</h3>
              <div className="metric-value">{metricsDisplay.classified}</div>
            </div>
            <div className="metric-card">
              <h3>Remaining</h3>
              <div className="metric-value">{metricsDisplay.remaining}</div>
            </div>
            <div className="metric-card">
              <h3>Progress</h3>
              <div className="metric-value">
                {((metricsDisplay.classified / (metricsDisplay.classified + metricsDisplay.remaining)) * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="single-point-container">
            {currentSinglePoint === null ? (
              <div className="empty-state">
                <p>No more points to classify or loading next point...</p>
                <button onClick={handleNextPoint} className="refresh-button">
                  Find Next Point
                </button>
              </div>
            ) : (
              <>
                <div className="single-point-card">
                  <h3>Point #{currentSinglePoint}</h3>
                  
                  <div className="point-visualization">
                    {/* Single point visualization */}
                    <Plot
                      data={[{
                        x: [hiddenReps[currentSinglePoint][0]],
                        y: [hiddenReps[currentSinglePoint][1]],
                        mode: 'markers',
                        type: 'scatter',
                        marker: {
                          size: 16,
                          color: getPointColor(currentSinglePoint),
                          line: {
                            color: 'black',
                            width: 2
                          }
                        }
                      }]}
                      layout={{
                        autosize: true,
                        title: 'Current Point',
                        xaxis: { 
                          title: 'First Principal Component',
                          gridcolor: '#e5e7eb',
                        },
                        yaxis: { 
                          title: 'Second Principal Component',
                          gridcolor: '#e5e7eb',
                        },
                        paper_bgcolor: 'white',
                        plot_bgcolor: 'white',
                        margin: { t: 60, r: 40, b: 60, l: 60 },
                      }}
                      config={{
                        responsive: true,
                        displayModeBar: false,
                        displaylogo: false,
                      }}
                      style={{ width: '100%', height: '300px' }}
                      useResizeHandler={true}
                    />
                  </div>
                  
                  <div className="point-details">
                    <div className="point-prediction">
                      <h4>Model Prediction</h4>
                      <div className="prediction-gauge">
                        <div 
                          className="prediction-fill" 
                          style={{ 
                            width: `${predictions[currentSinglePoint] * 100}%`,
                            backgroundColor: predictions[currentSinglePoint] > 0.7 ? COLORS.ATTACK : 
                                             predictions[currentSinglePoint] < 0.3 ? COLORS.BENIGN : 
                                             COLORS.UNCERTAIN 
                          }}
                        ></div>
                      </div>
                      <div className="prediction-value">
                        {(predictions[currentSinglePoint] * 100).toFixed(1)}% attack probability
                      </div>
                      <div className="prediction-label">
                        Model classifies as: <strong>{predictions[currentSinglePoint] >= 0.5 ? 'Attack' : 'Benign'}</strong>
                      </div>
                    </div>
                    
                    <div className="feature-analysis">
                      <h4>Top Features</h4>
                      <div className="feature-list">
                        {getPointTopFeatures(currentSinglePoint, 5).map((feature, idx) => (
                          <div key={idx} className="feature-item">
                            <div className="feature-name" data-name={feature.name}>{feature.name}</div>
                            <div className="feature-bar-container">
                              <div 
                                className={`feature-bar ${feature.value > 0 ? 'positive' : 'negative'}`}
                                style={{ width: `${Math.min(Math.abs(feature.importance) * 100, 100)}%` }}
                              ></div>
                            </div>
                            <div className="feature-value">
                              {feature.value.toFixed(4)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  <div className="single-point-actions">
                    <button 
                      onClick={() => handleSinglePointClassify(0)} 
                      disabled={loading}
                      className="classify-button benign"
                    >
                      Classify as Benign
                    </button>
                    <button 
                      onClick={() => handleSinglePointClassify(1)} 
                      disabled={loading}
                      className="classify-button attack"
                    >
                      Classify as Attack
                    </button>
                    <button 
                      onClick={handleNextPoint} 
                      disabled={loading}
                      className="skip-button"
                    >
                      Skip to Next
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>

          <button onClick={handleReset} className="reset-button">
            Reset Environment
          </button>
        </div>
      )}

      {activeTab === 'history' && (
        <div className="history-tab">
          <h2>Classification History</h2>
          {trainingHistory.length === 0 ? (
            <div className="empty-state">
              No classification history yet. Select points and classify them to build history.
            </div>
          ) : (
            <div className="history-list">
              <div className="history-header">
                <div className="history-cell">Point ID</div>
                <div className="history-cell">True Label</div>
                <div className="history-cell">Predicted</div>
                <div className="history-cell">Correct</div>
                <div className="history-cell">Reward</div>
              </div>
              {trainingHistory.map((entry, idx) => (
                <div key={idx} className={`history-row ${entry.correct ? 'correct' : 'incorrect'}`}>
                  <div className="history-cell">{entry.index}</div>
                  <div className="history-cell">{entry.true_label === 0 ? 'Benign' : 'Attack'}</div>
                  <div className="history-cell">{entry.predicted_label === 0 ? 'Benign' : 'Attack'}</div>
                  <div className="history-cell">{entry.correct ? '✓' : '✗'}</div>
                  <div className="history-cell">{entry.reward}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {activeTab === 'metrics' && (
        <div className="metrics-tab">
          <h2>Training Metrics Dashboard</h2>
          
          <div className="metrics-dashboard">
            <div className="metrics-chart">
              <h3>Accuracy Metrics</h3>
              <div className="metric-bars">
                <div className="metric-bar-item">
                  <div className="metric-label">Accuracy</div>
                  <div className="metric-bar-container">
                    <div 
                      className="metric-bar-baseline"
                      style={{ width: `70%` }}
                    ></div>
                    <div 
                      className="metric-bar-fill"
                      style={{ width: `${metrics ? metrics.accuracy * 100 : 0}%` }}
                    ></div>
                  </div>
                  <div className="metric-value">
                    <span className="base-value">70%</span>
                    <span className="current-value">{metricsDisplay.accuracy}%</span>
                  </div>
                </div>
                <div className="metric-bar-item">
                  <div className="metric-label">Precision</div>
                  <div className="metric-bar-container">
                    <div 
                      className="metric-bar-baseline"
                      style={{ width: `65%` }}
                    ></div>
                    <div 
                      className="metric-bar-fill"
                      style={{ width: `${metrics ? metrics.precision * 100 : 0}%` }}
                    ></div>
                  </div>
                  <div className="metric-value">
                    <span className="base-value">65%</span>
                    <span className="current-value">{metricsDisplay.precision}%</span>
                  </div>
                </div>
                <div className="metric-bar-item">
                  <div className="metric-label">Recall</div>
                  <div className="metric-bar-container">
                    <div 
                      className="metric-bar-baseline"
                      style={{ width: `75%` }}
                    ></div>
                    <div 
                      className="metric-bar-fill"
                      style={{ width: `${metrics ? metrics.recall * 100 : 0}%` }}
                    ></div>
                  </div>
                  <div className="metric-value">
                    <span className="base-value">75%</span>
                    <span className="current-value">{metricsDisplay.recall}%</span>
                  </div>
                </div>
                <div className="metric-bar-item">
                  <div className="metric-label">F1 Score</div>
                  <div className="metric-bar-container">
                    <div 
                      className="metric-bar-baseline"
                      style={{ width: `72%` }}
                    ></div>
                    <div 
                      className="metric-bar-fill"
                      style={{ width: `${metrics ? metrics.f1 * 100 : 0}%` }}
                    ></div>
                  </div>
                  <div className="metric-value">
                    <span className="base-value">72%</span>
                    <span className="current-value">{metricsDisplay.f1}%</span>
                  </div>
                </div>
              </div>
              <div className="metrics-legend">
                <div className="legend-item">
                  <div className="legend-color baseline"></div>
                  <div>Base Model</div>
                </div>
                <div className="legend-item">
                  <div className="legend-color current"></div>
                  <div>Current Performance</div>
                </div>
              </div>
            </div>
            
            <div className="confusion-matrix">
              <h3>Confusion Matrix</h3>
              {metrics && (
                <div className="matrix-container">
                  <div className="matrix-row">
                    <div className="matrix-cell header"></div>
                    <div className="matrix-cell header">Predicted Benign</div>
                    <div className="matrix-cell header">Predicted Attack</div>
                  </div>
                  <div className="matrix-row">
                    <div className="matrix-cell header">Actual Benign</div>
                    <div className="matrix-cell true-negative">{metrics.true_negatives}</div>
                    <div className="matrix-cell false-positive">{metrics.false_positives}</div>
                  </div>
                  <div className="matrix-row">
                    <div className="matrix-cell header">Actual Attack</div>
                    <div className="matrix-cell false-negative">{metrics.false_negatives}</div>
                    <div className="matrix-cell true-positive">{metrics.true_positives}</div>
                  </div>
                </div>
              )}
            </div>
            
            <div className="classification-stats">
              <h3>Classification Stats</h3>
              <div className="stats-grid">
                <div className="stat-item">
                  <div className="stat-label">Total Classified:</div>
                  <div className="stat-value">{metrics ? metrics.total_classified : 0}</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">Remaining:</div>
                  <div className="stat-value">{metrics ? metrics.remaining_unclassified : 0}</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">True Positives:</div>
                  <div className="stat-value">{metrics ? metrics.true_positives : 0}</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">True Negatives:</div>
                  <div className="stat-value">{metrics ? metrics.true_negatives : 0}</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">False Positives:</div>
                  <div className="stat-value">{metrics ? metrics.false_positives : 0}</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">False Negatives:</div>
                  <div className="stat-value">{metrics ? metrics.false_negatives : 0}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App; 