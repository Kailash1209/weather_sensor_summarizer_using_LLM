import tkinter as tk
from tkinter import font

class WeatherDisplay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🌀 Weather Summary Dashboard")
        self.root.geometry("700x300")
        self.root.configure(bg="#1e1e2f")

        self.title_font = font.Font(family="Helvetica", size=20, weight="bold")
        self.text_font = font.Font(family="Helvetica", size=14)

        self.temp_label = tk.Label(self.root, text="", font=self.text_font, fg="white", bg="#1e1e2f")
        self.hum_label = tk.Label(self.root, text="", font=self.text_font, fg="white", bg="#1e1e2f")
        self.pres_label = tk.Label(self.root, text="", font=self.text_font, fg="white", bg="#1e1e2f")
        self.summary_label = tk.Label(self.root, text="", font=self.text_font, fg="lightgreen", bg="#1e1e2f", wraplength=650, justify="left")

        tk.Label(self.root, text="🌦 Real-Time Weather Summary", font=self.title_font, fg="#00c0ff", bg="#1e1e2f").pack(pady=10)
        self.temp_label.pack(pady=5)
        self.hum_label.pack(pady=5)
        self.pres_label.pack(pady=5)
        self.summary_label.pack(pady=10)

        self.root.update()

    def update(self, temp, hum, pres, summary):
        self.temp_label.config(text=f"🌡 Temperature: {temp}°C")
        self.hum_label.config(text=f"💧 Humidity: {hum}%")
        self.pres_label.config(text=f"🌬 Pressure: {pres} hPa")
        self.summary_label.config(text=f"📝 Summary: {summary}")
        self.root.update()
