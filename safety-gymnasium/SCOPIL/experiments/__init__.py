import atexit
from SCOPIL.utils.display_utils import set_monitor, stop_virtual_monitor

process, display_id = set_monitor()

# At the exit stop the virtual display (in case of headless machine)
atexit.register(stop_virtual_monitor, process=process, display_id=display_id)