import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import random
from ortools.linear_solver import pywraplp

st.set_page_config(
    page_title="P-Median Problem Demo",
    layout="wide",
)

# with st.expander("What is the P-Median Problem?"):
#     st.markdown("""
#     ### The P-Median Problem
    
#     The p-median problem is a classic facility location problem that aims to:
#     - Select p facilities from a set of potential locations
#     - Assign demand points to their closest selected facility
#     - Minimize the total distance between demand points and their assigned facilities
#     """)

# Simplified sidebar controls
with st.sidebar:
    st.header("Parameters")
    num_points = st.slider("Number of demand points", 15, 30, 20)
    p = st.slider("Number of facilities (p)", 2, 5, 3)
    
    # Fixed geographic area (Turkey)
    center_lat, center_lon = 39.0, 35.0
    min_lat, max_lat = 37.0, 41.0
    min_lon, max_lon = 27.0, 43.0
    
    run_button = st.button("Generate Random Points and Solve")

# Function to generate random points with minimum distance constraint
def generate_random_points(num_points, min_lat, max_lat, min_lon, max_lon, min_distance=1):
    points = []
    while len(points) < num_points:
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        demand = random.randint(100, 1000)
        
        # Check if the new point is sufficiently far from existing points
        too_close = False
        for point in points:
            distance = np.sqrt((lat - point["latitude"])**2 + (lon - point["longitude"])**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            points.append({
                "ID": len(points) + 1,
                "latitude": lat,
                "longitude": lon,
                "demand": demand,
                "name": f"Point {len(points) + 1}"
            })
    
    return pd.DataFrame(points)

# Function to calculate distance matrix
def calculate_distance_matrix(df):
    n = len(df)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        lat1, lon1 = df.iloc[i]["latitude"], df.iloc[i]["longitude"]
        for j in range(n):
            lat2, lon2 = df.iloc[j]["latitude"], df.iloc[j]["longitude"]
            distance_matrix[i, j] = 100 * np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    
    return distance_matrix

# Function to solve the p-median problem
def solve_p_median(distance_matrix, demand, p):
    n = distance_matrix.shape[0]
    solver = pywraplp.Solver.CreateSolver("SCIP")
    
    # Variables
    x = {}  # x[i, j] = 1 if demand point j is assigned to facility i
    y = {}  # y[i] = 1 if facility is located at i
    
    for i in range(n):
        y[i] = solver.BoolVar(f"y[{i}]")
        for j in range(n):
            x[i, j] = solver.BoolVar(f"x[{i},{j}]")
    
    # Objective function: minimize total weighted distance
    objective = solver.Sum(distance_matrix[i, j] * demand[j] * x[i, j] for i in range(n) for j in range(n))
    solver.Minimize(objective)
    
    # Constraints
    # 1. Each demand point must be assigned to exactly one facility
    for j in range(n):
        solver.Add(solver.Sum(x[i, j] for i in range(n)) == 1)
    
    # 2. Demand points can only be assigned to selected facilities
    for i in range(n):
        for j in range(n):
            solver.Add(x[i, j] <= y[i])
    
    # 3. Exactly p facilities must be selected
    solver.Add(solver.Sum(y[i] for i in range(n)) == p)
    
    # Solve the model
    status = solver.Solve()
    
    if status == pywraplp.Solver.OPTIMAL:
        # Get the selected facilities
        selected_facilities = [i for i in range(n) if y[i].solution_value() > 0.5]
        
        # Get the assignments
        assignments = {}
        for j in range(n):
            for i in range(n):
                if x[i, j].solution_value() > 0.5:
                    assignments[j] = i
                    break
        
        # Calculate objective value
        objective_value = solver.Objective().Value()
        
        return selected_facilities, assignments, objective_value
    else:
        return None, None, None

# Main function - ONLY run when button is clicked or app initializes for the first time
if run_button or 'data' not in st.session_state:
    # Generate random points
    df = generate_random_points(num_points, min_lat, max_lat, min_lon, max_lon)
    
    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(df)

    demand = df["demand"].values
    
    # Solve p-median problem
    selected_facilities, assignments, objective_value = solve_p_median(distance_matrix, demand, p)
    
    # Store results in session state
    st.session_state.data = df
    st.session_state.distance_matrix = distance_matrix
    st.session_state.selected_facilities = selected_facilities
    st.session_state.assignments = assignments
    st.session_state.objective_value = objective_value

# Display results only if we have data
if 'data' in st.session_state:
    # Use results from session state
    df = st.session_state.data
    distance_matrix = st.session_state.distance_matrix
    selected_facilities = st.session_state.selected_facilities
    assignments = st.session_state.assignments
    objective_value = st.session_state.objective_value

    # Display results
    if selected_facilities:
        # Generate distinct colors for each facility
        colors = [
            [255, 0, 0],    # Red
            [0, 0, 255],    # Blue
            [0, 255, 0],    # Green
            [128, 0, 128],  # Purple
            [255, 165, 0],  # Orange
            [139, 0, 0],    # Dark Red
            [173, 216, 230],# Light Blue
            [95, 158, 160], # Cadet Blue
            [0, 0, 139],    # Dark Blue
            [0, 100, 0]     # Dark Green
        ]
        
        # Prepare data for PyDeck
        points_data = []
        for idx, row in df.iterrows():
            facility_idx = assignments[idx]
            color_idx = selected_facilities.index(facility_idx) % len(colors)
            
            point_type = "facility" if idx in selected_facilities else "demand"

            # Normalize demand values to a scale of 0.5 to 1.0
            min_demand = df["demand"].min()
            max_demand = df["demand"].max()
            
            # Set radius proportional to normalized demand
            radius = 0.1 + (row["demand"] - min_demand) / (max_demand - min_demand) * 0.9
            radius *= 50000  # Scale factor to adjust the size for visualization
            
            points_data.append({
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "demand": row["demand"],
                "name": row["name"],
                "point_type": point_type,
                "color": colors[color_idx],
                "radius": radius,
                "assigned_to": facility_idx + 1,
                "is_facility": idx in selected_facilities
            })
        
        points_df = pd.DataFrame(points_data)
        
        # Create PyDeck map
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=4.9,
            pitch=0
        )
        
        # Create layers
        scatterplot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=points_df,
            get_position=["longitude", "latitude"],
            get_radius="radius",
            get_fill_color="color",
            get_line_color=[0, 0, 0],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True
        )
        
        # Create paths (lines) between demand points and their assigned facilities
        paths_data = []
        for demand_idx, facility_idx in assignments.items():
            color_idx = selected_facilities.index(facility_idx) % len(colors)
            paths_data.append({
                "path": [
                    [df.iloc[demand_idx]["longitude"], df.iloc[demand_idx]["latitude"]],
                    [df.iloc[facility_idx]["longitude"], df.iloc[facility_idx]["latitude"]]
                ],
                "color": colors[color_idx]  # Use the corresponding facility color
            })
        
        # Create a DataFrame for paths
        paths_df = pd.DataFrame(paths_data)
        
        # Create a PathLayer for the lines
        path_layer = pdk.Layer(
            "PathLayer",
            data=paths_df,
            get_path="path",
            get_color="color",
            width_scale=20,
            width_min_pixels=2,
            pickable=False
        )
        

        # Add the PathLayer to the deck
        deck = pdk.Deck(
            layers=[scatterplot_layer, path_layer],
            initial_view_state=view_state,
            tooltip={
                "text": "{name}\nDemand: {demand}\nAssigned to: Facility {assigned_to}"
            }
        )

        
        # Show the map
        st.pydeck_chart(deck)

        ############# REPORTING #############
        
        # # Show results
        # st.markdown(f"**Total weighted distance:** {objective_value:.2f}")
        
        # # Display facilities and assignments
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     st.markdown("**Selected Facilities**")
        #     facility_df = pd.DataFrame([{
        #         "Facility ID": idx + 1,
        #         "Latitude": df.iloc[idx]["latitude"],
        #         "Longitude": df.iloc[idx]["longitude"],
        #         "Points Served": list(assignments.values()).count(idx)
        #     } for idx in selected_facilities])
        #     st.dataframe(facility_df)
        
        # with col2:
        #     st.markdown("**Assignment Summary**")
        #     assignment_summary = []
        #     for facility_idx in selected_facilities:
        #         assigned_points = [j for j, i in assignments.items() if i == facility_idx]
        #         total_demand = sum(df.iloc[point]["demand"] for point in assigned_points)
        #         assignment_summary.append({
        #             "Facility ID": facility_idx + 1,
        #             "Points Assigned": len(assigned_points),
        #             "Total Demand": total_demand,
        #             "Avg Distance": np.mean([distance_matrix[facility_idx, j] for j in assigned_points]) if assigned_points else 0
        #         })
        #     st.dataframe(pd.DataFrame(assignment_summary))
        
        # # Show all assignment details
        # with st.expander("Show All Assignment Details"):
        #     assignment_details = []
        #     for demand_idx, facility_idx in assignments.items():
        #         assignment_details.append({
        #             "Point ID": demand_idx + 1,
        #             "Assigned To": facility_idx + 1,
        #             "Distance": distance_matrix[facility_idx, demand_idx],
        #             "Demand": df.iloc[demand_idx]["demand"],
        #             "Weighted Distance": distance_matrix[facility_idx, demand_idx] * df.iloc[demand_idx]["demand"]
        #         })
        #     st.dataframe(pd.DataFrame(assignment_details))


        ################################ Reporting Section
else:
    # Display a message if no data is available yet
    st.info("Click 'Generate Random Points and Solve' to run the model.")
