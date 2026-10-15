UAV-to-Ground Channel Digital Twin — Python version

RUN IN VS CODE
1. Extract this folder.
2. Open the folder in VS Code.
3. Open Terminal in VS Code.
4. Create a virtual environment (optional but recommended):
   python -m venv .venv
   .venv\Scripts\activate       (Windows)
5. Install dependencies:
   pip install -r requirements.txt
6. Start the app:
   streamlit run app.py

The app opens in your browser. Use the sidebar to change UAV altitude, distance,
frequency, transmit power, environment, and training settings. Click "Train both
models" to train the DNN and PI-DNN and display the charts and metrics.

Note: This is a Python/Streamlit recreation of the supplied HTML dashboard. The
visual layout and interactive workflow are similar, but it is not pixel-identical.
