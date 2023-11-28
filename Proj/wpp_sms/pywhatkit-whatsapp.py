import pywhatkit
import datetime
import time
import threading

def auto_reply_whatsapp_message(message):
    """
    This function takes a message as an input and auto-replies to it like a human.
    
    Parameters:
    message (str): The message to be replied to.
    
    Returns:
    str: The auto-reply message.
    """
    
    # Import necessary libraries
    import re
    import random
    
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


while(datetime.datetime.now().hour != 18):
    # Obtém a hora atual
    agora = datetime.datetime.now()

    # Extrai horas, minutos e segundos
    horas = agora.hour
    minutos = agora.minute
    segundos = agora.second

    # phone_num = input("Enter your phone number: ")
    print(f"Agora são {horas} horas, {minutos} minutos e {segundos} segundos.")
    # mensagem = f"Agora são {horas} horas, {minutos} minutos e {segundos} segundos."


    mensagem = auto_reply_whatsapp_message("Ola?")

    # Calcula o tempo para enviar a mensagem (20 minutos após o horário atual)
    horas_envio = horas
    minutos_envio = minutos + 2

    # Verifica se os minutos ultrapassaram 60 e ajusta as horas e minutos correspondentes
    if minutos_envio >= 60:
        horas_envio += 1
        minutos_envio -= 60

    # After countdown, send the message
    pywhatkit.sendwhatmsg(
        phone_no="+351914411549",
        message=mensagem,
        time_hour=horas_envio,
        time_min=minutos_envio,
        wait_time=15,
        tab_close=False,
        close_time=3
    )

print("\nMessage sent successfully!")

