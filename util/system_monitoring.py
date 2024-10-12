import psutil
import time
import threading
import numpy as np
import matplotlib.pyplot as plt

stop_thread = False
cpu_util = []
memory_util = []
disk_read_bytes = []
disk_write_bytes = []
io_read_counts = []
io_write_counts = []
start_io_counters = None
start_time = 0


def write_to_file(data, filename="./report/output.txt"):
    with open(filename, 'a') as f:
        f.write(data + '\n')


def monitor_system_resources(interval=0.1):
    global start_io_counters, stop_thread
    start_io_counters = psutil.disk_io_counters()

    while not stop_thread:
        start_time = time.time()

        # CPU util
        cpu_percent = psutil.cpu_percent()
        cpu_util.append(cpu_percent)

        # RAM util
        memory = psutil.virtual_memory()
        memory_util.append(memory.percent)

        # IO statistics
        io_counters = psutil.disk_io_counters()
        disk_read_bytes.append(io_counters.read_bytes)
        disk_write_bytes.append(io_counters.write_bytes)
        io_read_counts.append(io_counters.read_count)
        io_write_counts.append(io_counters.write_count)

        elapsed_time = time.time() - start_time
        time.sleep(max(0, (interval - elapsed_time)))


def calculate_average_statistics(duration):
    global start_io_counters
    io_counters = psutil.disk_io_counters()

    write_to_file("\nAverage System Resource Utilization:")
    write_to_file(f"Average CPU usage: {np.mean(cpu_util):.5f}%")
    write_to_file(f"Average RAM usage: {np.mean(memory_util):.5f}%")

    total_read_mbs = (io_counters.read_bytes -
                      start_io_counters.read_bytes) // (1024 ** 2)
    total_write_mbs = (io_counters.write_bytes -
                       start_io_counters.write_bytes) // (1024 ** 2)
    write_to_file(
        f"Total Disk bytes - Read: {total_read_mbs}MB, Write: {total_write_mbs}MB")
    write_to_file(
        f"Average Disk MBs - Read: {total_read_mbs / duration:.5f}, Write: {total_write_mbs / duration:.5f}")

    total_io_read_counts = io_counters.read_count - start_io_counters.read_count
    total_io_write_counts = io_counters.write_count - start_io_counters.write_count
    write_to_file(
        f"Total IO - Read: {total_io_read_counts}, Write: {total_io_write_counts}")
    write_to_file(
        f"Average IOPS - Read: {total_io_read_counts / duration:.5f}, Write: {total_io_write_counts / duration:.5f}")

    busy_time_total = io_counters.busy_time - start_io_counters.busy_time
    disk_io_percentage = (busy_time_total / duration * 1000) * 100
    write_to_file(
        f"Disk I/O Busy Time Percentage: {disk_io_percentage:.5f}% of total execution time\n\n")


def get_plot(duration, fig_name):
    global cpu_util, memory_util, disk_read_bytes, disk_write_bytes, io_read_counts, io_write_counts
    disk_read_cumulative = (np.array(disk_read_bytes) -
                            disk_read_bytes[0]) // (1024 ** 2)
    disk_write_cumulative = (np.array(disk_write_bytes) - 
                            disk_write_bytes[0]) // (1024 ** 2)
    io_read_counts_cumulative = np.array(io_read_counts) - io_read_counts[0]
    io_write_counts_cumulative = np.array(io_write_counts) - io_write_counts[0]

    plt.figure(figsize=(12, 10))

    # CPU utilization plot
    plt.subplot(3, 2, 1)
    plt.plot(np.linspace(0, duration * 10, len(cpu_util)),
             cpu_util, label='CPU Utilization (%)', color='b')
    plt.xlabel('Time (0.1 seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Utilization Over Time')
    plt.ylim(0, 100)
    plt.grid(True)

    # Memory utilization plot
    plt.subplot(3, 2, 2)
    plt.plot(np.linspace(0, duration * 10, len(memory_util)),
             memory_util, label='RAM Utilization (%)', color='g')
    plt.xlabel('Time (0.1 seconds)')
    plt.ylabel('RAM Usage (%)')
    plt.title('RAM Utilization Over Time')
    plt.grid(True)

    # Disk Read MBs (Cumulative)
    plt.subplot(3, 2, 3)
    plt.plot(np.linspace(0, duration * 10, len(disk_read_cumulative)),
             disk_read_cumulative, label='Disk Read (MB)', color='r')
    plt.xlabel('Time (0.1 seconds)')
    plt.ylabel('Disk Read (MB)')
    plt.title('Disk Read Over Time')
    plt.ylim(bottom=0)
    plt.grid(True)

    # Disk Write MBs (Cumulative)
    plt.subplot(3, 2, 4)
    plt.plot(np.linspace(0, duration * 10, len(disk_write_cumulative)),
             disk_write_cumulative, label='Disk Write (MB)', color='c') 
    plt.xlabel('Time (0.1 seconds)')
    plt.ylabel('Disk Write (MB)')
    plt.title('Disk Write Over Time')
    plt.ylim(bottom=0)
    plt.grid(True)

    # IO Read
    plt.subplot(3, 2, 5)
    plt.plot(np.linspace(0, duration * 10, len(io_read_counts_cumulative)),
             io_read_counts_cumulative, label='# IO Read Counts', color='m')
    plt.xlabel('Time (0.1 seconds)')
    plt.ylabel('# IO Read Counts')
    plt.title('# IO Read Counts Over Time')
    plt.ylim(bottom=0)
    plt.grid(True)

    # IO Write
    plt.subplot(3, 2, 6)
    plt.plot(np.linspace(0, duration * 10, len(io_write_counts_cumulative)),
             io_write_counts_cumulative, label='# IO Write Counts', color='y')
    plt.xlabel('Time (0.1 seconds)')
    plt.ylabel('# IO Write Counts')
    plt.title('# IO Write Counts Over Time')
    plt.ylim(bottom=0)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"./img/{fig_name}.png")


def start_monitoring():
    global start_time
    start_time = time.time()
    monitor_thread = threading.Thread(
        target=monitor_system_resources, args=(10,), daemon=True)
    monitor_thread.start()
    return monitor_thread


def stop_monitoring(monitor_thread, fig_name="system_monitoring"):
    global start_time, cpu_util, memory_util, disk_read_bytes, disk_write_bytes, io_read_counts, io_write_counts, stop_thread
    stop_thread = True
    monitor_thread.join(timeout=1)
    duration = time.time() - start_time
    write_to_file(f"Execution Time of {fig_name}: {duration:.5f} seconds")
    calculate_average_statistics(duration)
    get_plot(duration, fig_name)

    cpu_util = []
    memory_util = []
    disk_read_bytes = []
    disk_write_bytes = []
    io_read_counts = []
    io_write_counts = []
    start_time = 0
    stop_thread = False


if __name__ == "__main__":
    monitor_thread = start_monitoring()
    time.sleep(600)
    stop_monitoring(monitor_thread)
