
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket

#Setup UDP communication parameters
UDP_IP = "172.26.228.242"
UDP_PORT = 25000

# Create the socket for the UDP communication
sock = socket.socket(socket.AF_INET,    # Family of addresses, in this case IP type 
                     socket.SOCK_DGRAM) # What protocol to use, in this case UDP (datagram)
sock.bind((UDP_IP, UDP_PORT))
print("Listening on IP:", UDP_IP, "Port:", UDP_PORT)

while True:
    #Read data
    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
    # Print data
    print ("received message:", data.decode('utf-8')) # As a string (check the ASCII table)
    if data.decode('utf-8') == "ENDSTOP":
        print("Endstop reached at x=0")
        break