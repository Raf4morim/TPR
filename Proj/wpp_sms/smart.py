import requests
import time
import random
import threading

def send_message(chat_id, text):
    """
    Send message to chat_id.
    :param chat_id: Phone number + "@c.us" suffix - 1231231231@c.us
    :param text: Message for the recipient
    """
    # Send a text message via WhatsApp HTTP API
    response = requests.post(
        "http://localhost:3000/api/sendText",
        json={
            "chatId": chat_id,
            "text": text,
            "session": "default",
        },
    )
    response.raise_for_status()

# List of contacts to send messages to
contacts = [
    "351914411549"
    # "351911781175",
    # "351915086600",
]

# Function to wait for user input
def wait_for_input():
    input("Press Enter to stop sending messages...\n")
    print("Stopping message sending...")
    # Terminate the program after Enter is pressed
    raise SystemExit

# Start a thread to wait for user input in the background
input_thread = threading.Thread(target=wait_for_input)
input_thread.daemon = True  # Mark the thread as a daemon so it can be stopped
input_thread.start()

message_counter = 0

while True:
    if not input_thread.is_alive():
        break

    if message_counter == 10:
        pause_time = int(random.uniform(10, 20))  # Random pause time between 10 to 20 seconds
        print(f"Pausing for {pause_time} seconds...")
        time.sleep(pause_time)
        message_counter = 0  # Reset the counter after the pause

    contact = random.choice(contacts)
    short_message_length = random.randint(3, 9)
    medium_message_length = random.randint(10, 49)
    long_message_length = random.randint(50, 100)
    
    # Choose between short and long message lengths with different weights
    message_length = random.choices([short_message_length, medium_message_length, long_message_length], weights=[0.7, 0.2, 0.1])[0] # mais provavel 1 pessoa escrever pouco do que mt xD
    
    # Create a message with random characters of the defined length
    message = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=message_length))
    # Send the message to the selected contact
    send_message(chat_id=f"{contact}@c.us", text=message)
    message_counter += 1
    
    # Set the time interval based on the message length
    if message_length <= 10:
        time.sleep(random.uniform(1, 3))
    elif message_length > 11 and message_length <= 40 :
        time.sleep(random.uniform(4, 10))  
    else:
        time.sleep(random.uniform(11, 15))
