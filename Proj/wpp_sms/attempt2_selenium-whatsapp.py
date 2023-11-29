import datetime
import time
import random
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

def auto_reply_whatsapp_message(message):
    """
    This function takes a message as an input and auto-replies to it like a human.
    
    Parameters:
    message (str): The message to be replied to.
    
    Returns:
    str: The auto-reply message.
    """
    
    # Define a list of possible responses
    responses = [
        "Hey there!",
        "What's up?",
        "How can I help you?",
        "What do you need?",
        "What can I do for you?",
        "What do you want to know?",
        "What can I do to help you?",
        "What can I do?",
        "What do you need help with?",
        "What do you want to talk about?"
    ]
    
    # Check if the message is a question
    if re.search(r'\?$', message):
        # If it is a question, return a random response from the list
        return random.choice(responses)
    else:
        # If it is not a question, return a generic response
        return "I see. Let me know if you need help with anything else."

# Define a function to send WhatsApp message using Selenium
def send_whatsapp_message(phone_no, message):
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("user-data-dir=./selenium")
        print("passou1")
        driver = webdriver.Chrome(service=Service("./chromedriver"), options=options)
        print("passou1")
        driver.get("https://web.whatsapp.com/send?phone=" + phone_no)
        print("passou1")
        time.sleep(10)  # Waiting for the page to load
        
        input_box = driver.find_element(By.XPATH, "//div[@contenteditable='true']")
        input_box.send_keys(message)
        input_box.send_keys(Keys.ENTER)
        
        time.sleep(5)  # Wait for the message to be sent
        
        driver.quit()
        
        print("Message sent successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

while(datetime.datetime.now().hour != 18): # Enquanto não forem 18 hrs
    # Get current time
    agora = datetime.datetime.now()

    # Extract hours, minutes, and seconds
    horas = agora.hour
    minutos = agora.minute
    segundos = agora.second

    print(f"Agora são {horas} horas, {minutos} minutos e {segundos} segundos.")

    mensagem = auto_reply_whatsapp_message("Ola?")

    # Calculate time to send the message (20 minutes after the current time)
    horas_envio = horas
    minutos_envio = minutos + 2

    if minutos_envio >= 60:
        horas_envio += 1
        minutos_envio -= 60

    # After countdown, send the message
    send_whatsapp_message("+351914411549", mensagem)

print("\nMessages sent successfully!")
