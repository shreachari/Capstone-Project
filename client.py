# This code is for the server 
# Lets import the libraries
import socket, cv2, pickle, struct, imutils, json, sys

# Socket Create
CLIENT_SOCKET = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
HOST_IP  = '3.129.7.97'  # Use the client's IP
# HOST_IP = '127.0.0.1'
# HOST_IP = '192.168.86.28'
PORT = 5050
SOCKET_ADDRESS = (HOST_IP,PORT)

# Socket Connect
CLIENT_SOCKET.connect(SOCKET_ADDRESS)



keep_going = True
got_user_info = False
client_socket = None
# Socket Accept
while keep_going:
    try:
        if not got_user_info:
            # Prompt user for input
            first_name = input("Enter your first name: ").strip()
            last_name = input("Enter your last name: ").strip()
            middle_name = input("Enter your middle name or press ENTER if none: ").strip()

            # Send user input to client
            CLIENT_SOCKET.sendall(json.dumps({"first_name": first_name, "last_name": last_name, "middle_name": middle_name}).encode())

            # Receive response from client
            data = CLIENT_SOCKET.recv(1024)
            if not data:
                continue
            if data.decode() == "Fail":
                print("User not found. Terminating connection.")
                CLIENT_SOCKET.close()
                sys.exit()
            else:
                got_user_info = True
                print(got_user_info)
        else:
            vid = cv2.VideoCapture(0)
            
            while(vid.isOpened()):
                img,frame = vid.read()
                if img:
                    frame = imutils.resize(frame,width=320)
                    a = pickle.dumps(frame)
                    message = struct.pack("Q",len(a))+a
                    
                    try:
                        CLIENT_SOCKET.sendall(message)
                    except socket.error as e:
                        CLIENT_SOCKET.close()
                        print("Client socket has closed.")
                        keep_going = False
                        break
                    
                    cv2.imshow('TRANSMITTING VIDEO',frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key ==ord('q'):
                        CLIENT_SOCKET.close()
    except socket.error as e:
        print("Socket error:", e)
        CLIENT_SOCKET.close()
        break
    except KeyboardInterrupt:
        CLIENT_SOCKET.close()
        print("Server stopped by KeyboardInterrupt")
        break
