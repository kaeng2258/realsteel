import serial

def connect_bluetooth(bluetooth_port='COM13', baud_rate=115200):
    try:
        bt_socket = serial.Serial(bluetooth_port, baud_rate)
        if not bt_socket.isOpen():
            bt_socket.open()
        print("Bluetooth serial port connected.")
        return bt_socket
    except Exception as e:
        print(f"Error: Could not open Bluetooth serial port. {e}")
        exit()

def send_robot_command(bt_socket, no):
    exeCmd = [0xff, 0xff, 0x4c, 0x53, 0x00, 0x00, 0x00, 0x00, 0x30, 0x0c, 0x03, 0x01, 0x00, 100, 0x00]
    exeCmd[11] = no
    exeCmd[14] = sum(exeCmd[6:14]) & 0xFF
    bt_socket.write(bytearray(exeCmd))
    print(f"Sent command: {exeCmd}")  # 디버깅 출력을 추가하여 전송된 명령 확인
