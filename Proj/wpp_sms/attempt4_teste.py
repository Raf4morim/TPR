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

# Lista de contatos para enviar mensagens
contacts = [
    "351914411549"
    # "351911781175",
    # "351915086600",
]

# Lista de expressões diferentes
expressions = [
    "Olá!",
    "Como vai?",
    "Tudo bem?",
    "Bom dia!",
    "Boa tarde!",
]

# Função para aguardar a entrada do usuário
def wait_for_input():
    input("Pressione Enter para parar o envio de mensagens...\n")
    print("Parando o envio de mensagens...")
    # Encerrar o programa após o Enter ser pressionado
    raise SystemExit

# Iniciar uma thread para aguardar a entrada do usuário em segundo plano
input_thread = threading.Thread(target=wait_for_input)
input_thread.daemon = True  # Marcar a thread como um daemon para que ela possa ser interrompida
input_thread.start()

while True:
    # Verificar se a thread de entrada do usuário está viva
    if not input_thread.is_alive():
        break  # Se a thread não estiver viva, interromper o loop
        
    # Selecionar aleatoriamente um contato da lista
    contact = random.choice(contacts)
    
    # Selecionar aleatoriamente uma expressão da lista
    expression = random.choice(expressions)
    
    # Gerar um tamanho aleatório para a mensagem entre 5 e 20 caracteres
    message_length = random.randint(5, 20)
    
    # Criar uma mensagem com a expressão e tamanho aleatórios
    message = f"{expression} {''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=message_length))}"
    
    # Enviar a mensagem para o contato selecionado
    send_message(chat_id=f"{contact}@c.us", text=message)
    
    # Gerar um atraso aleatório entre 5 e 15 segundos
    delay = random.randint(1, 5)
    
    # Aguardar o atraso antes de enviar a próxima mensagem
    time.sleep(delay)
