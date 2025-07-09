import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import random
from time import time

class SensorMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor Monitor")
        self.root.geometry("1000x700")
        
        # Data storage
        self.timestamps = []
        self.pressure_data = []
        self.temperature_data = []
        self.humidity_data = []
        self.max_points = 100
        
        # Create main layout
        self.create_figure()
        self.create_controls()
        
        # Start animation
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=500)
        
    def create_figure(self):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.5)
        
        # Pressure plot
        self.pressure_line, = self.ax1.plot([], [], 'r-')
        self.ax1.set_title('Pressure (hPa)')
        self.ax1.set_ylim(980, 1050)
        self.ax1.grid(True)
        
        # Temperature plot
        self.temp_line, = self.ax2.plot([], [], 'b-')
        self.ax2.set_title('Temperature (°C)')
        self.ax2.set_ylim(10, 40)
        self.ax2.grid(True)
        
        # Humidity plot
        self.humidity_line, = self.ax3.plot([], [], 'g-')
        self.ax3.set_title('Humidity (%)')
        self.ax3.set_ylim(0, 100)
        self.ax3.grid(True)
        
        # Embed in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_controls(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(control_frame, text="Update Interval (ms):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.StringVar(value="500")
        ttk.Entry(control_frame, textvariable=self.interval_var, width=6).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Apply", command=self.update_interval).pack(side=tk.LEFT, padx=10)
        ttk.Button(control_frame, text="Clear Data", command=self.clear_data).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Export Data", command=self.export_data).pack(side=tk.RIGHT)
        
        # Live value display
        self.live_values = ttk.Frame(self.root)
        self.live_values.pack(fill=tk.X, padx=10, pady=5)
        
        self.pressure_label = ttk.Label(self.live_values, text="Pressure: -- hPa")
        self.temp_label = ttk.Label(self.live_values, text="Temperature: -- °C")
        self.hum_label = ttk.Label(self.live_values, text="Humidity: -- %")
        
        self.pressure_label.pack(side=tk.LEFT, padx=20)
        self.temp_label.pack(side=tk.LEFT, padx=20)
        self.hum_label.pack(side=tk.LEFT, padx=20)
    
    def read_sensor(self):
        """Simulate sensor readings or connect to real sensors here"""
        current_time = time()
        
        # Simulated data (replace with actual sensor readings)
        pressure = 1013 + random.uniform(-5, 5) + 10 * np.sin(current_time/10)
        temperature = 25 + random.uniform(-1, 1) + 5 * np.sin(current_time/15)
        humidity = 50 + random.uniform(-3, 3) + 20 * np.sin(current_time/20)
        
        return pressure, temperature, humidity
    
    def update_plot(self, frame):
        # Get new sensor data
        pressure, temperature, humidity = self.read_sensor()
        current_time = time()
        
        # Update data lists
        self.timestamps.append(current_time)
        self.pressure_data.append(pressure)
        self.temperature_data.append(temperature)
        self.humidity_data.append(humidity)
        
        # Maintain fixed length
        if len(self.timestamps) > self.max_points:
            self.timestamps.pop(0)
            self.pressure_data.pop(0)
            self.temperature_data.pop(0)
            self.humidity_data.pop(0)
        
        # Convert to relative time
        rel_time = [t - self.timestamps[0] for t in self.timestamps]
        
        # Update plots
        self.pressure_line.set_data(rel_time, self.pressure_data)
        self.temp_line.set_data(rel_time, self.temperature_data)
        self.humidity_line.set_data(rel_time, self.humidity_data)
        
        # Update axis limits
        for ax, data in zip([self.ax1, self.ax2, self.ax3], 
                           [self.pressure_data, self.temperature_data, self.humidity_data]):
            if len(rel_time) > 0:
                ax.set_xlim(0, max(rel_time))
        
        # Update live values
        self.pressure_label.config(text=f"Pressure: {pressure:.1f} hPa")
        self.temp_label.config(text=f"Temperature: {temperature:.1f} °C")
        self.hum_label.config(text=f"Humidity: {humidity:.1f} %")
        
        return self.pressure_line, self.temp_line, self.humidity_line
    
    def update_interval(self):
        try:
            interval = int(self.interval_var.get())
            self.animation.event_source.interval = interval
        except ValueError:
            pass
    
    def clear_data(self):
        self.timestamps.clear()
        self.pressure_data.clear()
        self.temperature_data.clear()
        self.humidity_data.clear()
    
    def export_data(self):
        try:
            with open('sensor_data.csv', 'w') as f:
                f.write("Timestamp,Pressure (hPa),Temperature (°C),Humidity (%)\n")
                for t, p, temp, h in zip(self.timestamps, self.pressure_data, 
                                         self.temperature_data, self.humidity_data):
                    f.write(f"{t},{p},{temp},{h}\n")
            print("Data exported to sensor_data.csv")
        except Exception as e:
            print(f"Export error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorMonitor(root)
    root.mainloop()
