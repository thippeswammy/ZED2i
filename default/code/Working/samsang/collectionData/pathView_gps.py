import matplotlib.pyplot as plt
import pandas as pd

# Load your CSV
gps_data = pd.read_csv('./Data/Session/gps_data.csv')
plt.plot(gps_data['longitude'], gps_data['latitude'], '-o')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('GPS Path')
plt.grid(True)
plt.show()
