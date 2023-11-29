import requests

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

# Definir o número de telefone do contato e a mensagem a ser enviada
contact_number = "351914411549"  # Substitua pelo número de telefone do seu contato
message = "Ueleleeee2"

# Enviar a mensagem para o contato específico
send_message(chat_id=f"{contact_number}@c.us", text=message)
