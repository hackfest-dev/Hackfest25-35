# ⚡ EcoVolt

EcoVolt is an AI-powered dashboard designed to help users forecast solar and wind energy investments based on real-time weather data, geographic location, and budget constraints.
This dashboard provides energy generation forecasts using machine learning models trained on historical data.

Key Features
Real-time weather-based energy prediction
Location & budget input
Interactive visualizations (Plotly charts, 3D Pydeck maps)

Tech stack:
Backend: Python, Streamlit, SQLite
Machine Learning: XGBoost, sklearn
Visualization: Plotly
Containerization: Docker
Frontend: NextJS

Installation and Setup
1. Clone the repository: git clone https://github.com/hackfest-dev/Hackfest25-35
2. Build the Docker image: cd energy-forecast-dashboard docker build -t energy-forecast-dashboard .
3. Run the Docker container: docker run -p 8501:8501 energy-forecast-dashboard

Frontend (Next.js)
npm install
npm run dev

The application will be available at http://localhost:8501.

Usage:
1. Choose the energy type you want to view (wind, solar, or demand) by clicking the corresponding button in the Live Prediction Demo section.
2. Select the Location and Budget.
3. Observe the real-time predictions and insights provided by the dashboard.
4. Utilize the chart visualizations to make informed decisions about energy strategies and resource potential.

   



