import os
import subprocess
import time
import signal


def check_monitor_connected():
    # This command checks for connected monitors
    try:
        output = subprocess.run(['xrandr'], capture_output=True, text=True)
        return ' connected ' in output.stdout
    except FileNotFoundError:
        print(
            "To allow monitor checking please run the following:"
            "\nsudo apt-get install x11-xserver-utils"
        )
        exit(0)


def install_xvfb():
    # Check if Xvfb is installed
    result = subprocess.run(['which', 'Xvfb'], capture_output=True, text=True)
    if not result.stdout.strip():
        print(
            "To allow a virtual monitor to be set up, please run the following:"
            "\nsudo apt-get install xvfb"
        )
        exit(0)


def start_xvfb():
    display_id = 1
    display = f':{display_id}'
    lock_file = f'/tmp/.X{display_id}-lock'
    socket_file = f'/tmp/.X11-unix/X{display_id}'

    # Find the first available virtual display
    while os.path.exists(lock_file) or os.path.exists(socket_file):
        display_id += 1
        display = f':{display_id}'
        lock_file = f'/tmp/.X{display_id}-lock'
        socket_file = f'/tmp/.X11-unix/X{display_id}'

        # Wait a bit to ensure everything is cleaned up
        time.sleep(2)

    # Start Xvfb
    print("Starting Xvfb...")
    process = subprocess.Popen(['Xvfb', display, '-screen', '0', '1024x768x16'])
    os.environ['DISPLAY'] = display
    print(f"DISPLAY set to {display}\n")

    return process, display_id


def set_monitor():
    process, display_id = None, None
    if not check_monitor_connected():
        print("No monitor detected. Setting up virtual display...")
        install_xvfb()
        process, display_id = start_xvfb()
    else:
        print("Monitor detected. Proceeding normally...")

    return process, display_id


def stop_virtual_monitor(process, display_id):
    if process is not None:
        print("Stopping Xvfb...")
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)  # Set a timeout to prevent hanging
        except subprocess.TimeoutExpired:
            print("Xvfb did not stop in time; forcing termination.")
            process.kill()  # Force kill if not stopped after timeout
        finally:
            print("Xvfb stopped successfully.")
            process = None

        assert display_id is not None, "While the virtual display process is not None, the 'display_id' is None!"
        lock_file = f'/tmp/.X{display_id}-lock'
        socket_file = f'/tmp/.X11-unix/X{display_id}'
        if os.path.exists(lock_file):
            print(f"Deleting lock file: {lock_file} ...")
            os.remove(lock_file)
            print(f"Deleted lock file.")
        if os.path.exists(socket_file):
            print(f"Deleting lock file: {socket_file} ...")
            os.remove(socket_file)
            print(f"Deleted socket file.")
