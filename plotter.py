import numpy as np
import time
import matplotlib.pyplot as plt
import threading
import queue


class DoaPlotter:
    def __init__(self) -> None:
        self.queue = queue.Queue()

        self.theta_history = []
        self.r_history = []

    def create_plot(self):
        plt.ion()  # interactive mode

        fig = plt.figure()
        self.ax = fig.add_subplot(111, polar=True)
        self.line, = self.ax.plot([], [], 'o-', label="Estimated DOA")

        self.ax.set_theta_zero_location('N')  # 0 deg at the top (North)
        self.ax.set_theta_direction(-1)       # angles increase clockwise
        self.ax.set_rmax(1)
        self.ax.set_rticks([])  # hide radial ticks
        self.ax.set_title("DOA Estimate (Theta)")

    def plot_thread(self):
        # Initialize polar plot
        self.create_plot()

        while True:
            # Get all queued DOA angles
            while not self.queue.empty():
                theta_deg = self.queue.get_nowait()
                theta_rad = np.deg2rad(theta_deg)

                self.theta_history.append(theta_rad)
                self.r_history.append(1)

                self.theta_history = self.theta_history[-2:]
                self.r_history = self.r_history[-2:]
                self.line.set_data(self.theta_history, self.r_history)

            self.ax.relim()
            self.ax.autoscale_view()

            plt.draw()
            plt.pause(0.01)
            time.sleep(0.01)

    def start(self):
        threading.Thread(
            target=self.plot_thread,
            daemon=True
        ).start()

    def put(self, value):
        self.queue.put(value)
