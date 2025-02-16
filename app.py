import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import numpy as np

st.markdown(
    """
    <style>
    /* General app styling */
    .stApp {
        background-image: url("https://wallpapercave.com/wp/wp7335231.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white !important;
    }
    /* Universal text color override */
    body, div, p, span, label, h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    /* Task option buttons styling */
    .stButton > button {
        background-color: green !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        padding: 10px 20px !important;
    }
    .stButton > button:hover {
        background-color: darkgreen !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: black !important;
        color: white !important;
    }
    /* Sidebar title */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
        color: white !important;
    }
    /* Sidebar option buttons */
    [data-testid="stSidebar"] .stRadio > div > label {
        color: white !important;
    }
    /* Input parameter blocks (dropdowns, select boxes) text color */
    .stSelectbox, .stDropdown, .stMultiselect {
        color: black !important;
    }
    .stTextInput input, .stTextArea textarea {
        color: black !important;
        background-color: white !important;
    }
    /* Slider styling */
    .stSlider .st-br {
        background-color: black !important;
        border-radius: 5px !important;
    }
    /* Ensure sidebar is visible and black on mobile devices */
    @media only screen and (max-width: 768px) {
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: black !important;
            display: block !important;
            position: fixed !important;
            width: 250px !important;
            height: 100vh !important;
            overflow-y: auto !important;
            z-index: 1000 !important;
        }
        /* Adjust the main content to avoid overlap with the sidebar */
        .css-12oz5g7 {
            margin-left: 260px !important; /* Make space for the fixed sidebar */
        }
        /* Ensure input text is black on mobile */
        .stSelectbox div, .stDropdown div, .stMultiselect div {
            color: black !important;
        }
        .stTextInput input, .stTextArea textarea {
            color: black !important;
            background-color: white !important;
        }
    }
    </style>
    """, unsafe_allow_html=True
)



# Initialize the Groq client with the API key from Streamlit's secrets
# Initialize the Groq client with the API key from Streamlit's secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
# Function to interact with the Groq API
def get_groq_response(user_input, model="deepseek-r1-distill-llama-70b"):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model=model
    )
    return chat_completion.choices[0].message.content
# Function to recommend panel type, power, and battery
def recommend_panel_and_battery():
    st.title("🌞 Solar Panel and Battery Recommendation")
    st.markdown("""
    Enter your home or office power requirements to get recommendations for the most suitable solar panel type, required power, and battery capacity.
    """)

    # Input Parameters
    st.header("🏠 Power Requirements")
    rooms = st.number_input("Number of Rooms", value=2, step=1)
    fans = st.number_input("Number of Fans", value=4, step=1)
    lights = st.number_input("Number of Lights", value=8, step=1)
    appliances_power = st.number_input("Other Appliances Power Consumption (Watts)", value=500, step=50)

    # Calculate Total Power Requirement
    fan_power = fans * 70  # Average power consumption per fan: 70W
    light_power = lights * 10  # Average power consumption per light: 10W
    total_power = rooms * (fan_power + light_power) + appliances_power  # Total power in watts
    total_power_kw = total_power / 1000  # Convert to kilowatts

    st.write(f"**Total Power Requirement**: {total_power_kw:.2f} kW")

    # Recommend Solar Panel Type and Power
    st.header("🔧 Recommended Solar Panel")
    if total_power_kw <= 1:
        panel_type = "Monocrystalline"
        panel_area = 10  # Assume 10 m² for small systems
    elif total_power_kw <= 3:
        panel_type = "Polycrystalline"
        panel_area = 30  # Assume 30 m² for medium systems
    else:
        panel_type = "Thin-Film"
        panel_area = 50  # Assume 50 m² for larger systems

    st.write(f"**Recommended Solar Panel Type**: {panel_type}")
    st.write(f"**Estimated Panel Area**: {panel_area} m²")

    # Recommend Battery Capacity
    st.header("🔋 Recommended Battery Capacity")
    battery_hours = 6  # Assume 6 hours of backup required
    battery_capacity = total_power_kw * battery_hours
    st.write(f"**Recommended Battery Capacity**: {battery_capacity:.2f} kWh")

# Function to calculate solar energy
def calculate_solar_energy():
    st.title("🔄 Solar Energy System Design and Analysis")
    st.markdown("""
    This application helps in designing and analyzing solar energy systems.
    Provide system specifications and site details to calculate potential power generation and energy storage.
    """)

    # Input Parameters
    st.header("🔧 Design Parameters")
    panel_type = st.selectbox("Select Solar Panel Type", ["Monocrystalline", "Polycrystalline", "Thin-Film"])
    
    panel_efficiency = st.number_input("Panel Efficiency (%)", value=18.0, step=0.1)
    st.markdown("""
    **Formula**: Panel Efficiency (%) = (Panel Power (kW) / Panel Area (m²)) x 100
    Panel efficiency refers to the percentage of sunlight converted into usable electricity by the panel.
    """)
    
    battery_capacity = st.number_input("Battery Storage Capacity (kWh)", value=10.0, step=0.5)
    st.markdown("""
    **Formula**: Battery Storage Capacity (kWh) = Energy Stored (kWh)
    This is the total energy a battery can store for later use.
    """)

    battery_efficiency = st.number_input("Battery Efficiency (%)", value=90.0, step=0.5)
    st.markdown("""
    **Formula**: Battery Efficiency (%) = (Energy Discharged / Energy Charged) x 100
    This refers to how efficiently energy can be discharged from the battery compared to how much energy was charged.
    """)

    tilt_angle = st.slider("Optimal Tilt Angle (Degrees)", min_value=0, max_value=45, value=30)
    st.markdown("""
    **Tilt Angle** refers to the angle at which solar panels are installed to maximize energy absorption.
    """)

    solar_insolation = st.number_input("Solar Insolation (kWh/m²/day)", value=5.5, step=0.1)
    st.markdown("""
    **Solar Insolation** is the amount of solar energy received per unit area per day. This varies depending on geographic location.
    """)

    area = st.number_input("Total Panel Area (m²)", value=50.0, step=1.0)
    st.markdown("""
    **Panel Area** refers to the total surface area of the solar panels installed.
    """)

    degradation_rate = st.number_input("Panel Degradation Rate (% per year)", value=0.5, step=0.1)
    st.markdown("""
    **Degradation Rate** refers to the percentage by which the panel's efficiency decreases over time.
    """)

    dust_loss = st.slider("Dust Loss Factor (%)", min_value=0, max_value=10, value=5)
    st.markdown("""
    **Dust Loss** is the percentage reduction in panel efficiency due to dust accumulation.
    """)

    shading_loss = st.slider("Shading Loss Factor (%)", min_value=0, max_value=10, value=3)
    st.markdown("""
    **Shading Loss** is the percentage reduction in energy generation due to partial shading of the panels.
    """)

    if st.button("Calculate Solar Energy"):
        # Calculations
        effective_area = area * (1 - (dust_loss + shading_loss) / 100)
        daily_energy = solar_insolation * effective_area * (panel_efficiency / 100)
        annual_energy = daily_energy * 365 * (1 - degradation_rate / 100)
        battery_energy = battery_capacity * (battery_efficiency / 100)

        st.header("✨ Calculated Results")
        st.write(f"Daily Energy Generation: {daily_energy:.2f} kWh")
        st.write(f"Annual Energy Generation (First Year): {annual_energy:.2f} kWh")
        st.write(f"Battery Storage Capacity: {battery_energy:.2f} kWh")

        # Explanation for results using Groq
        explanation_input = f"Explain the solar energy system results based on the following values: Daily Energy Generation = {daily_energy:.2f} kWh, Annual Energy Generation = {annual_energy:.2f} kWh, Battery Storage = {battery_energy:.2f} kWh."
        explanation = get_groq_response(explanation_input)
        st.markdown(f"### Detailed Explanation: {explanation}")

        # Visualization: Seasonal Power Generation
        months = np.arange(1, 13)
        seasonal_insolation = np.array([
            solar_insolation * (1 + 0.1 * np.sin((month - 1) * np.pi / 6)) for month in months
        ])
        monthly_energy = seasonal_insolation * effective_area * (panel_efficiency / 100) * 30

        fig, ax = plt.subplots()
        ax.plot(months, monthly_energy, label='Monthly Energy Generation (kWh)', color='orange')
        ax.set_xlabel('Month')
        ax.set_ylabel('Energy (kWh)')
        ax.set_title('Seasonal Power Generation')
        ax.legend()
        st.pyplot(fig)

        # Visualization: Battery Storage Performance
        time = np.linspace(0, 24, 100)
        usage_pattern = battery_energy * (1 - 0.05 * np.sin(time * np.pi / 12))

        fig2, ax2 = plt.subplots()
        ax2.plot(time, usage_pattern, label='Battery Performance (kWh)', color='blue')
        ax2.set_xlabel('Time (Hours)')
        ax2.set_ylabel('Energy Stored (kWh)')
        ax2.set_title('Battery Storage Over a Day')
        ax2.legend()
        st.pyplot(fig2)
# Function to generate system design for deep-sea tidal energy systems
def generate_system_design():
    st.title("⚙️ Deep-Sea Tidal Energy System Design")
    st.markdown("""
    This application helps design deep-sea tidal energy systems using cutting-edge materials and advanced design techniques.
    You will input various parameters related to materials, depth, and tidal velocity, and we will generate optimized system designs.
    """)

    # Inputs for system design
    st.header("📊 Input Parameters")
    material = st.selectbox("🛠️ Select Material for System Components", 
                            ["Titanium Alloys (e.g., Ti-6Al-4V)", 
                             "Fiber-Reinforced Polymers (FRP)", 
                             "Cermets", 
                             "Advanced Coatings"])
    depth = st.number_input("🌊 Enter Depth (meters)", min_value=100, max_value=5000, step=100)
    tidal_velocity = st.number_input("💨 Enter Tidal Velocity (m/s)", min_value=0.1, max_value=10.0, step=0.1)
    biofouling_control = st.selectbox("🌱 Select Biofouling Control Strategy", 
                                      ["Fluoropolymers", 
                                       "Ultrasonic Cleaning Systems", 
                                       "Biocidal Coatings",
                                       "Self-Cleaning Coatings",
                                       "Electrochemical Anti-Fouling",
                                       "Mechanical Cleaning Systems"])

    location = st.selectbox("📍 Select Location for Tidal System", ["Tropical Ocean", "Temperate Ocean", "Polar Ocean"])
    if location == "Tropical Ocean":
        water_temperature = st.slider("🌡️ Water Temperature (°C)", min_value=25, max_value=30, value=28)
        salinity = st.slider("🌊 Salinity (ppt)", min_value=30, max_value=40, value=35)
        tidal_pattern = st.selectbox("🌊 Tidal Pattern", ["Semi-diurnal", "Diurnal"])
    elif location == "Temperate Ocean":
        water_temperature = st.slider("🌡️ Water Temperature (°C)", min_value=10, max_value=20, value=15)
        salinity = st.slider("🌊 Salinity (ppt)", min_value=20, max_value=30, value=25)
        tidal_pattern = st.selectbox("🌊 Tidal Pattern", ["Mixed", "Semi-diurnal"])
    else:
        water_temperature = st.slider("🌡️ Water Temperature (°C)", min_value=-2, max_value=10, value=5)
        salinity = st.slider("🌊 Salinity (ppt)", min_value=30, max_value=40, value=35)
        tidal_pattern = st.selectbox("🌊 Tidal Pattern", ["Diurnal", "Mixed"])

    environmental_sensitivity = st.selectbox("🌎 Select Environmental Sensitivity", 
                                            ["Protected Ecosystem", "Unprotected Ecosystem"])

    if st.button("🔍 Generate System Design"):
        user_input = f"Design a deep-sea tidal energy system for the following parameters: Material: {material}, Depth: {depth}m, Tidal Velocity: {tidal_velocity}m/s, Biofouling Control: {biofouling_control}, Location: {location}, Water Temperature: {water_temperature}°C, Salinity: {salinity}ppt, Tidal Pattern: {tidal_pattern}, Environmental Sensitivity: {environmental_sensitivity}."
        system_design = get_groq_response(user_input)
        st.header("✨ Generated System Design")
        st.write(system_design)

        # Adding Visualization: A simple line chart to visualize input parameters
        input_params = ['Depth', 'Tidal Velocity', 'Water Temp', 'Salinity']
        input_values = [depth, tidal_velocity, water_temperature, salinity]
        
        # Plotting a line chart for input parameters
        fig2, ax2 = plt.subplots()
        ax2.plot(input_params, input_values, marker='o', color='purple')
        ax2.set_title('Tidal Energy System Design Inputs')
        ax2.set_ylabel('Value')
        st.pyplot(fig2)

# Function to calculate power generation for tidal plants
def calculate_power_generation():
    st.title("⚡ Power Generation Calculation for Tidal Plant")
    st.markdown("""
    This application calculates the potential power generation of a tidal plant.
    Formula used:
    
    P = 1/2 * ρ * A * v^3 * Cₑ
    Where:
    - P: Power (Watts)
    - ρ: Water density (kg/m³), typically 1025 kg/m³ for seawater
    - A: Area swept by turbine blades (m²)
    - v: Tidal current velocity (m/s)
    - Cₑ: Efficiency coefficient (dimensionless)
    """)

    # Input parameters
    water_density = st.number_input("💧 Enter Water Density (kg/m³)", value=1025, step=1)
    swept_area = st.number_input("⚙️ Enter Swept Area of Turbine Blades (m²)", value=1000, step=10)
    velocity = st.number_input("💨 Enter Tidal Current Velocity (m/s)", value=2.0, step=0.1)
    efficiency = st.number_input("⚡ Enter Efficiency Coefficient (0 to 1)", value=0.4, step=0.01)

    if st.button("🔢 Calculate Power"):
        power = 0.5 * water_density * swept_area * (velocity ** 3) * efficiency
        st.header("✨ Calculated Power Output")
        st.write(f"The potential power generation is {power:.2f} Watts.")

        # Adding Visualization: Displaying power as a curve chart
        velocities = np.linspace(0, velocity, 100)
        powers = 0.5 * water_density * swept_area * (velocities ** 3) * efficiency

        fig3, ax3 = plt.subplots()
        ax3.plot(velocities, powers, color='green')
        ax3.set_title('Power Generation Curve')
        ax3.set_xlabel('Tidal Current Velocity (m/s)')
        ax3.set_ylabel('Power (Watts)')
        st.pyplot(fig3)

# Function to generate corrosion-resistant coating suggestions
def generate_coating_suggestions():
    st.title("🛡️ Corrosion-Resistant Coating Suggestions for Deep-Sea Tidal Energy Systems")
    st.markdown("""
    This application helps suggest the most suitable corrosion-resistant coatings for deep-sea tidal energy systems.
    Input various environmental conditions and system material, and we will recommend the best coating to ensure system longevity.
    """)

    # Inputs for Environmental Conditions and Material Type
    st.header("🌊 Input Environmental Conditions and Material Type")
    salinity = st.slider("🌊 Salinity (ppt)", min_value=20, max_value=40, value=35, step=1)
    temperature = st.slider("🌡️ Temperature (°C)", min_value=-10, max_value=40, value=20, step=1)
    wave_force = st.slider("💨 Wave and Current Forces (0: Low, 10: High)", min_value=0, max_value=10, value=5)
    uv_exposure = st.slider("☀️ UV Exposure (0: Low, 10: High)", min_value=0, max_value=10, value=5)
    material_type = st.selectbox("🛠️ Select Material Type", 
                                 ["Titanium Alloys (e.g., Ti-6Al-4V)", 
                                  "Stainless Steel", 
                                  "Aluminum Alloys", 
                                  "Fiber-Reinforced Polymers (FRP)", 
                                  "Other"])

    if st.button("🔍 Suggest Coating"):
        user_input = f"Suggest a corrosion-resistant coating for a deep-sea tidal energy system with the following parameters: Salinity: {salinity}ppt, Temperature: {temperature}°C, Wave and Current Forces: {wave_force}/10, UV Exposure: {uv_exposure}/10, Material Type: {material_type}."
        coating_suggestion = get_groq_response(user_input)
        st.header("✨ Recommended Corrosion-Resistant Coating")
        st.write(coating_suggestion)

        # Adding Visualization: A simple bar chart of the factors for better understanding
        factors = ["Salinity", "Temperature", "Wave and Current Forces", "UV Exposure"]
        values = [salinity, temperature, wave_force, uv_exposure]

        # Plotting a bar chart for input factors
        fig, ax = plt.subplots()
        ax.bar(factors, values, color='skyblue')
        ax.set_xlabel('Factors')
        ax.set_ylabel('Value')
        ax.set_title('Corrosion-Resistant Coating Factors')
        ax.set_xticklabels(factors, rotation=45, ha='right')  # Avoid overlap by rotating the labels
        st.pyplot(fig)

# Wind power calculation function
def calculate_power(wind_speed, blade_length, efficiency):
    air_density = 1.225  # kg/m^3
    swept_area = np.pi * (blade_length ** 2)
    power = 0.5 * air_density * swept_area * (wind_speed ** 3) * (efficiency / 100)
    return power / 1000  # Convert to kW

# Function to plot wind profile
def plot_wind_profile(heights, wind_speeds):
    df = pd.DataFrame({'Height': heights, 'Wind Speed': wind_speeds})
    fig = px.line(
        df, 
        x='Wind Speed', 
        y='Height', 
        markers=True, 
        line_shape='linear', 
        title="Wind Profile Analysis"
    )
    fig.update_traces(
        line=dict(color='#0000FF', dash='solid'), 
        marker=dict(size=10, symbol='circle')
    )
    fig.update_layout(
        xaxis_title="Wind Speed (m/s)", 
        yaxis_title="Height (m)"
    )
    st.plotly_chart(fig)

# Wind power calculator UI
def wind_power_calculator():
    st.subheader("Wind Power Calculator")
    st.markdown("""
    Calculate wind power output based on key inputs:
    - **Wind Speed** (m/s)
    - **Blade Length** (m)
    - **Efficiency** (%)
    """)

    # Inputs for calculation
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0, max_value=30, value=12)
    blade_length = st.number_input("Blade Length (m)", min_value=1, max_value=100, value=50)
    efficiency = st.number_input("Efficiency (%)", min_value=1, max_value=100, value=85)

    # Calculate and display power output
    power_output = calculate_power(wind_speed, blade_length, efficiency)
    st.write(f"**Calculated Power Output:** {power_output:.2f} kW")

    # Plot power vs wind speed
    wind_speeds = np.linspace(0, 30, 100)
    powers = [calculate_power(ws, blade_length, efficiency) for ws in wind_speeds]

    # Using seaborn for better aesthetics
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=wind_speeds, y=powers, ax=ax, label=f"Blade: {blade_length}m, Eff: {efficiency}%")
    ax.set_title("Power Output vs Wind Speed", fontsize=16)
    ax.set_xlabel("Wind Speed (m/s)", fontsize=12)
    ax.set_ylabel("Power Output (kW)", fontsize=12)
    ax.legend(fontsize=10)
    st.pyplot(fig)
def turbine_recommendation_system():
    """
    Single function to manage the Hydro-River Turbine Recommendation System, including API setup, querying, 
    and Streamlit UI interaction.
    """

    # Example Preloaded Text (Simulating PDF Content)
    PRELOADED_TEXT = """
    Hydropower turbines are categorized based on head and flow rate. For a head range of 10–20 meters, Kaplan turbines are suitable, 
    whereas Pelton turbines work best for heads above 50 meters. Flow rates also play a significant role; high-flow, low-head applications 
    favor Francis turbines. Additional factors to consider when choosing a turbine include the specific design and efficiency, as well as 
    site-specific conditions such as environmental impact, cost, and operational requirements.
    """

    # Step 2: Query System
    def query_system(user_input):
        # Use Groq API for response
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="llama-3.3-70b-versatile",
        )
        return response.choices[0].message.content

    # Step 3: Turbine Suggestion Logic
    def turbine_suggestion(head, flow_rate, turbine_design, efficiency, site_conditions):
        query = f"I have a head of {head} meters and a flow rate of {flow_rate} L/s. What turbine should I use?"
        response = query_system(query)

        # Add additional turbine specifications to the response
        additional_info = (
            f"\n\n📌 To make a more informed decision, consider additional factors such as the specific design and efficiency of the turbines."
            f"\n✔️ Selected Design: {turbine_design}"
            f"\n✔️ Efficiency: {efficiency}"
            f"\n✔️ Site Conditions: {site_conditions}"
            "\n🌍 Site-specific conditions like environmental impact, cost, and operational requirements also play a significant role."
        )

        return response + additional_info

    # Step 4: Streamlit UI
    st.title("⚙️ Hydro-River Turbine Recommendation System")
    st.write(
        "🌟 Welcome to the Turbine Recommendation System! 🌟\n\n"
        "💡 Select the **head**, **flow rate**, and other factors like **turbine design**, **efficiency**, and **site conditions** "
        "to receive expert turbine recommendations tailored to your parameters.\n"
        "🛠️ Powered by AI."
    )

    # Dropdown inputs for the user
    head_options = [10, 20, 30, 40, 50, 100]
    flow_rate_options = [100, 200, 300, 400, 500, 1000]
    turbine_design_options = ["Kaplan", "Pelton", "Francis", "Mixed Design"]
    efficiency_options = ["High", "Medium", "Low"]
    site_conditions_options = ["Environmental Impact", "Cost", "Operational Requirements", "All of the Above"]

    head = st.selectbox("💧 Select Head (meters)", head_options)
    flow_rate = st.selectbox("🌊 Select Flow Rate (L/s)", flow_rate_options)
    turbine_design = st.selectbox("🔧 Select Turbine Design", turbine_design_options)
    efficiency = st.selectbox("⚡ Select Efficiency Level", efficiency_options)
    site_conditions = st.selectbox("🌍 Select Site Conditions", site_conditions_options)

    if st.button('Get Turbine Suggestion'):
        result = turbine_suggestion(head, flow_rate, turbine_design, efficiency, site_conditions)
        st.subheader("Recommended Turbine:")
        st.write(result)


import streamlit as st

import streamlit as st

# Main function
def main():
    st.sidebar.title("🌊🌍🌊 BluePlanet Energy")
    
    # Add a detailed description to the main screen
    st.title("🌍 BluePlanet Energy Application")
    st.write("""
    This **Renewable Energy System Application** is designed to assist engineers, researchers, and enthusiasts 
    in evaluating and designing renewable energy systems. Whether you're working with solar, tidal, wind, or hydro energy, 
    this tool can provide insights and recommendations to optimize energy production and system performance.""")

    # Add options to the sidebar for selecting tasks
    option = st.sidebar.radio(
        "Choose Task", 
        ["Solar Energy Calculation", "Recommend Solar Panel and Battery", "Tidal System Design", 
         "Tidal Power Calculation", "Coating Suggestions for Tidal System", "Wind Power Calculator", 
         "Hydro-River Turbine Recommendation"]
    )

    # Perform task based on the selected option
    if option == "Solar Energy Calculation":
        calculate_solar_energy()
    elif option == "Recommend Solar Panel and Battery":
        recommend_panel_and_battery()
    elif option == "Tidal System Design":
        generate_system_design()
    elif option == "Tidal Power Calculation":
        calculate_power_generation()
    elif option == "Coating Suggestions for Tidal System":
        generate_coating_suggestions()
    elif option == "Wind Power Calculator":
        wind_power_calculator()
    elif option == "Hydro-River Turbine Recommendation":
        turbine_recommendation_system()

if __name__ == "__main__":
    main()
