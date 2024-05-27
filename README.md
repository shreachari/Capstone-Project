# Capstone-Project

## Overview

This repository consists of Shrea Chari's Capstone Project for the UCLA M.S. Computer Science degree. It contains code to run an application which checks the identity of a user based on a provided user profile. The application requires user input in the form of text, head movements, and finger movements.

## Setup

1. Clone the repository
2. Ensure python is installed
3. Install all dependencies by running ```pip install -r requirements.txt```
4. Ensure docker is installed and the daemon is running
5. Open two terminal windows and ensure you are in the repository folder
6. In one window run the following commands:
    a. ```docker build -t capstone .```
    b. ```docker run -p 5050:5050 -it capstone```
7. In the other window, run the command ```python client.py```
8. Follow the prompts

## Important notes

- You will need to ensure that the 'HOST_IP' variable in the file client.py is set to the IP address where the server will be run. 

- The port number can be changed. Ensure that it is changed in the files: client.py, server.py, and Dockerfile. The port number must be changed in the command ```docker run -p 5050:5050 -it capstone``` as well.
