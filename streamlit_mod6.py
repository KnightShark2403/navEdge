import streamlit as st
import requests
import datetime
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2

# Set page config
st.set_page_config(
    page_title="Travel Time Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
"""<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #333 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .info-box {
        background-color: #FFC0CB;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
    }
    .warning-box {
        background-color: #FFF3E0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #FF9800;
    }
    .success-box {
        background-color: #E8F5E9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>"""
, unsafe_allow_html=True)

# API Configuration
API_KEYS = {
    'here': "INX9nRlj31wrSl3jwNbOg1SbZkFVuyEaNpDaEM9AGyo",
    'tomtom': "hTb1sD4CQ9BS1gezCl1xjwGWQgZX4Iax",
    'osrm': None
}

# Historical traffic patterns
TRAFFIC_PATTERNS = {
    'day_patterns': {
        0: 1.2406, 1: 1.2344, 2: 1.2274, 3: 1.1883, 
        4: 1.2241, 5: 1.2285, 6: 1.2416
    },
    'hour_patterns': {
        0: 0.7, 1: 0.6, 2: 0.5, 3: 0.5, 4: 0.6, 5: 0.8,
        6: 1.1, 7: 1.4, 8: 1.5, 9: 1.3, 10: 1.2, 11: 1.1,
        12: 1.1, 13: 1.2, 14: 1.2, 15: 1.3, 16: 1.4,
        17: 1.6, 18: 1.5, 19: 1.3, 20: 1.1, 21: 0.9,
        22: 0.8, 23: 0.7
    }
}

class TravelTimePredictor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.api_status = {
            'here': {'working': True, 'last_error': None},
            'tomtom': {'working': True, 'last_error': None},
            'osrm': {'working': True, 'last_error': None}
        }
        
    def _get_here_data(self, origin, destination, departure):
        """Get HERE API data"""
        try:
            params = {
                "apiKey": API_KEYS['here'],
                "origin": f"{origin[0]},{origin[1]}",
                "destination": f"{destination[0]},{destination[1]}",
                "departureTime": departure.isoformat(timespec='seconds'),
                "return": "summary",
                "routeAttributes": "summary",
                "trafficMode": "enabled",
                "alternatives": 0
            }
            response = requests.get(
                "https://router.hereapi.com/v8/routes",
                params=params,
                timeout=5
            )
            
            if response.status_code == 401:
                self.api_status['here']['working'] = False
                self.api_status['here']['last_error'] = "Invalid API key"
                return None
                
            response.raise_for_status()
            data = response.json()
            
            if 'routes' not in data or not data['routes']:
                return None
                
            section = data['routes'][0]['sections'][0]
            return {
                'time': section['summary']['duration'] / 60,
                'distance': section['summary']['length'] / 1000,
                'source': 'here'
            }
            
        except Exception as e:
            self.api_status['here']['last_error'] = str(e)
            return None

    def _get_tomtom_data(self, origin, destination, departure):
        """Get TomTom API data"""
        try:
            depart_at = departure.isoformat(timespec='seconds') + 'Z'
            
            url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin[0]},{origin[1]}:{destination[0]},{destination[1]}/json"
            params = {
                "key": API_KEYS['tomtom'],
                "traffic": "true",
                "departAt": depart_at,
                "routeType": "fastest",
                "travelMode": "car"
            }
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code in [403, 401]:
                self.api_status['tomtom']['working'] = False
                self.api_status['tomtom']['last_error'] = "Invalid API key"
                return None
                
            response.raise_for_status()
            data = response.json()
            
            if 'routes' not in data or not data['routes']:
                return None
                
            route = data['routes'][0]
            return {
                'time': route['summary']['travelTimeInSeconds'] / 60,
                'distance': route['summary']['lengthInMeters'] / 1000,
                'source': 'tomtom'
            }
            
        except Exception as e:
            self.api_status['tomtom']['last_error'] = str(e)
            return None

    def _get_osrm_data(self, origin, destination):
        """Get OSRM data"""
        try:
            url = f"http://router.project-osrm.org/route/v1/driving/{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
            params = {
                "overview": "false",
                "alternatives": "false"
            }
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code != 200:
                self.api_status['osrm']['working'] = False
                self.api_status['osrm']['last_error'] = f"HTTP {response.status_code}"
                return None
                
            data = response.json()
            
            if 'routes' not in data or not data['routes']:
                return None
                
            route = data['routes'][0]
            return {
                'time': route['duration'] / 60,
                'distance': route['distance'] / 1000,
                'source': 'osrm'
            }
            
        except Exception as e:
            self.api_status['osrm']['last_error'] = str(e)
            return None

    def _get_traffic_multiplier(self, departure_time):
        """Get traffic multiplier based on historical patterns"""
        day_of_week = departure_time.weekday()
        hour_of_day = departure_time.hour
        
        day_multiplier = TRAFFIC_PATTERNS['day_patterns'].get(day_of_week, 1.0)
        hour_multiplier = TRAFFIC_PATTERNS['hour_patterns'].get(hour_of_day, 1.0)
        
        return day_multiplier * hour_multiplier

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate straight-line distance between coordinates"""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def predict(self, origin, destination, departure=None):
        """Get travel time prediction using all available APIs"""
        if not departure:
            departure = datetime.datetime.now()
        
        traffic_multiplier = self._get_traffic_multiplier(departure)
        
        # Get data from all APIs in parallel
        futures = []
        if self.api_status['here']['working']:
            futures.append(self.executor.submit(self._get_here_data, origin, destination, departure))
        if self.api_status['tomtom']['working']:
            futures.append(self.executor.submit(self._get_tomtom_data, origin, destination, departure))
        if self.api_status['osrm']['working']:
            futures.append(self.executor.submit(self._get_osrm_data, origin, destination))
        
        results = []
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
        
        # Calculate prediction
        if results:
            avg_time = mean([r['time'] for r in results])
            avg_distance = mean([r['distance'] for r in results])
            adjusted_time = avg_time * traffic_multiplier
            
            return {
                'distance': round(avg_distance, 1),
                'base_time': round(avg_time, 1),
                'adjusted_time': round(adjusted_time, 1),
                'traffic_multiplier': round(traffic_multiplier, 3),
                'confidence': 'high' if len(results) > 1 else 'medium',
                'sources': [r['source'] for r in results],
                'data_points': len(results),
                'day_of_week': departure.strftime('%A'),
                'hour_of_day': departure.hour
            }
        else:
            # Fallback to historical patterns
            distance = self._haversine_distance(origin[0], origin[1], destination[0], destination[1])
            base_time = distance * 2  # 30 km/h average speed
            adjusted_time = base_time * traffic_multiplier
            
            return {
                'distance': round(distance, 1),
                'base_time': round(base_time, 1),
                'adjusted_time': round(adjusted_time, 1),
                'traffic_multiplier': round(traffic_multiplier, 3),
                'confidence': 'low (historical only)',
                'sources': [],
                'data_points': 0,
                'day_of_week': departure.strftime('%A'),
                'hour_of_day': departure.hour,
                'warning': 'API services unavailable. Using historical patterns only.'
            }

def create_traffic_pattern_chart():
    """Create a chart showing traffic patterns by hour"""
    hours = list(range(24))
    multipliers = [TRAFFIC_PATTERNS['hour_patterns'][h] for h in hours]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=multipliers,
        mode='lines+markers',
        name='Traffic Multiplier',
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=8, color='#1565C0')
    ))
    
    fig.update_layout(
        title='Traffic Patterns by Hour of Day',
        xaxis_title='Hour of Day',
        yaxis_title='Traffic Multiplier',
        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        height=400,
        template='plotly_white',
    )
    
    return fig

def create_days_of_week_chart():
    """Create a chart showing traffic patterns by day of week"""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_values = [TRAFFIC_PATTERNS['day_patterns'][i] for i in range(7)]
    
    colors = ['#1E88E5' if x < 1.23 else '#FF9800' for x in day_values]
    
    fig = go.Figure(go.Bar(
        x=days,
        y=day_values,
        marker_color=colors,
        text=[f"{x:.2f}" for x in day_values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Traffic Patterns by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Traffic Multiplier',
        height=400,
        template='plotly_white',
    )
    
    return fig

def create_route_map(origin, destination):
    """Create a map showing the route"""
    fig = go.Figure()
    
    # Add the straight-line route
    fig.add_trace(go.Scattermapbox(
        mode = "markers+lines",
        lon = [origin[1], destination[1]],
        lat = [origin[0], destination[0]],
        marker = {'size': 10, 'color': ['green', 'red']},
        line = {'width': 4, 'color': '#1E88E5'},
        name = "Route"
    ))
    
    # Set map center at midpoint
    center_lat = (origin[0] + destination[0]) / 2
    center_lon = (origin[1] + destination[1]) / 2
    
    fig.update_layout(
        mapbox={
            'style': "open-street-map",
            'center': {'lon': center_lon, 'lat': center_lat},
            'zoom': 11
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=500
    )
    
    return fig

def create_confidence_gauge(confidence_level):
    """Create a gauge chart for confidence level"""
    confidence_value = 0.9 if confidence_level == "high" else 0.6 if confidence_level == "medium" else 0.3
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence_value * 100,
        title = {'text': "Prediction Confidence"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [0, 33], 'color': "rgba(244, 67, 54, 0.2)"},
                {'range': [33, 66], 'color': "rgba(255, 152, 0, 0.2)"},
                {'range': [66, 100], 'color': "rgba(76, 175, 80, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 0.75,
                'value': confidence_value * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    
    return fig

def main():
    st.markdown("<h1 class='main-header'>🚗 Travel Time Predictor</h1>", unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = TravelTimePredictor()
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>📍 Location Settings</h2>", unsafe_allow_html=True)
        
        location_type = st.radio(
            "Input Mode",
            ["Example Location", "Custom Coordinates"],
            key="location_type"
        )
        
        if location_type == "Example Location":
            origin = (12.9079, 77.6801)  # Sarjapur
            destination = (12.9235, 77.6862)  # RMZ Ecoworld
            st.info("Using example: Sarjapur to RMZ Ecoworld, Bangalore")
        else:
            st.markdown("#### Origin Coordinates")
            origin_lat = st.number_input("Start Latitude", value=12.9079)
            origin_lon = st.number_input("Start Longitude", value=77.6801)
            
            st.markdown("#### Destination Coordinates")
            dest_lat = st.number_input("End Latitude", value=12.9235)
            dest_lon = st.number_input("End Longitude", value=77.6862)
            
            origin = (origin_lat, origin_lon)
            destination = (dest_lat, dest_lon)
        
        st.markdown("<h2 class='sub-header'>⏰ Departure Settings</h2>", unsafe_allow_html=True)
        
        time_type = st.radio(
            "Departure Time",
            ["Now", "Custom Time"],
            key="time_type"
        )
        
        if time_type == "Now":
            departure = datetime.datetime.now()
            st.info(f"Using current time: {departure.strftime('%Y-%m-%d %H:%M')}")
        else:
            departure_date = st.date_input("Select Date", datetime.date.today())
            departure_time = st.time_input("Select Time", datetime.time(9, 0))
            departure = datetime.datetime.combine(departure_date, departure_time)
        
        # API Selection
        st.markdown("<h2 class='sub-header'>🔌 API Settings</h2>", unsafe_allow_html=True)
        use_here = st.checkbox("Use HERE API", value=True)
        use_tomtom = st.checkbox("Use TomTom API", value=True)
        use_osrm = st.checkbox("Use OSRM API", value=True)
        
        # Update API status based on user selection
        predictor.api_status['here']['working'] = use_here
        predictor.api_status['tomtom']['working'] = use_tomtom
        predictor.api_status['osrm']['working'] = use_osrm
        
        if st.button("🚀 Generate Prediction", use_container_width=True):
            st.session_state.predict = True
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 class='sub-header'>📊 Traffic Patterns</h2>", unsafe_allow_html=True)
        st.plotly_chart(create_traffic_pattern_chart(), use_container_width=True)
        st.plotly_chart(create_days_of_week_chart(), use_container_width=True)
    
    with col2:
        st.markdown("<h2 class='sub-header'>🗺️ Route Map</h2>", unsafe_allow_html=True)
        st.plotly_chart(create_route_map(origin, destination), use_container_width=True)
    
    # Generate prediction when button is clicked
    if st.session_state.get('predict', False):
        with st.spinner("Fetching data from APIs..."):
            prediction = predictor.predict(origin, destination, departure)
        
        st.markdown("<h2 class='sub-header'>🔮 Prediction Results</h2>", unsafe_allow_html=True)
        
        # Display warning if present
        if 'warning' in prediction:
            st.markdown(f"""
            <div class='warning-box'>
                <h3>⚠️ Warning</h3>
                <p>{prediction['warning']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{prediction['distance']} km</div>
                <div class='metric-label'>Distance</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{prediction['base_time']} min</div>
                <div class='metric-label'>Base Time</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{prediction['adjusted_time']} min</div>
                <div class='metric-label'>Adjusted Time</div>
            </div>
            """, unsafe_allow_html=True)
            
            speed = prediction['distance'] / (prediction['adjusted_time']/60)
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{speed:.1f} km/h</div>
                <div class='metric-label'>Avg Speed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{prediction['traffic_multiplier']}</div>
                <div class='metric-label'>Traffic Multiplier</div>
            </div>
            """, unsafe_allow_html=True)
            
            confidence_color = "#4CAF50" if prediction['confidence'].startswith("high") else "#FF9800" if prediction['confidence'].startswith("medium") else "#F44336"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color: {confidence_color}'>{prediction['confidence']}</div>
                <div class='metric-label'>Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Traffic conditions assessment
        if speed < 20:
            st.markdown("""
            <div class='warning-box'>
                <h3>⚠️ Heavy Traffic Expected</h3>
                <p>Average speed is below 20 km/h. Expect significant delays.</p>
            </div>
            """, unsafe_allow_html=True)
        elif 20 <= speed < 30:
            st.markdown("""
            <div class='warning-box'>
                <h3>⚠️ Moderate Traffic</h3>
                <p>Average speed is between 20-30 km/h. Some congestion likely.</p>
            </div>
            """, unsafe_allow_html=True)
        elif 30 <= speed < 40:
            st.markdown("""
            <div class='success-box'>
                <h3>✅ Normal Traffic Conditions</h3>
                <p>Average speed is between 30-40 km/h. Normal flow expected.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='success-box'>
                <h3>✅ Clear Traffic Conditions</h3>
                <p>Average speed is above 40 km/h. Smooth journey expected.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed results
        st.markdown("<h2 class='sub-header'>🔍 Detailed Information</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Timing Information")
            st.markdown(f"**Day of Week:** {prediction['day_of_week']}")
            st.markdown(f"**Hour of Day:** {prediction['hour_of_day']}:00")
            st.markdown(f"**Departure Time:** {departure.strftime('%Y-%m-%d %H:%M')}")
            
            st.markdown("#### Data Sources")
            if prediction['sources']:
                st.success(f"Data received from: {', '.join(prediction['sources']).upper()}")
            else:
                st.warning("No API data received - using historical patterns")
            
        with col2:
            st.plotly_chart(create_confidence_gauge(prediction['confidence'].split()[0]), use_container_width=True)
            
            st.markdown("#### API Status")
            for api, status in predictor.api_status.items():
                if status['working']:
                    st.success(f"{api.upper()}: Enabled")
                else:
                    st.error(f"{api.upper()}: Disabled")
                if status['last_error']:
                    st.warning(f"Last error: {status['last_error']}")

if __name__ == "__main__":
    main()