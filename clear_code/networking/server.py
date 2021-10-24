import socket
import time



def run(settings_map, introduce=False):

    # host_ip = settings_map['model_holders_ip']
    host_ip = settings_map['my_private_ip']
    host_port = int(settings_map['model_holders_port'])

    # print(host_ip)
    # print(host_port)

    others_metadata = None
    addr = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        if introduce:
            print("Connecting to Alice...")
            time.sleep(1)

        s.bind((host_ip, host_port))
        s.listen(1)  # We only want to connect with one person
        conn, addr = s.accept()
        with conn:
            if introduce:
                print("Connected to Alice - Hi Alice!")
                time.sleep(1)
            #print('Connected by', addr)
            while True:
                data = conn.recv(1024).decode()
                if others_metadata is None:
                    others_metadata = data
                if not data:
                    break

    # print(others_metadata)
    # addr[0] is the ip
    return others_metadata, addr[0]
