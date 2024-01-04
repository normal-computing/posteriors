def parse_devices(devices):
    devices = devices.split(",")
    devices_list = []
    for device in devices:
        try:
            device = int(device)
        except:
            pass
        devices_list.append(device)
    return devices_list
