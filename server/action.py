import socket
import pyautogui

# 定义监听的主机和端口
HOST = '0.0.0.0'  # 监听所有可用的接口
PORT = 5000

# 创建一个套接字对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 将套接字绑定到指定的主机和端口
s.bind((HOST, PORT))

# 开始监听端口
s.listen(1)

print(f"监听端口 {PORT} 中...")

while True:
    # 等待客户端连接
    conn, addr = s.accept()
    print(f"接收到来自 {addr[0]} 的连接")

    # 触发Ctrl + Win + 左箭头键以切换虚拟桌面
    pyautogui.hotkey('ctrl', 'winleft', 'left')
    response = "HTTP/1.1 200 OK\r\n\r\n"
    conn.send(response.encode())
    # 关闭连接
    conn.close()
