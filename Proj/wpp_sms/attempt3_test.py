from whatsapp_api_client_python import API

greenAPI = API.GreenAPI(
    "1101000001", "d75b3a66374942c5b3c019c698abc2067e151558acbd412345"
)



def main():
    response = greenAPI.sending.sendMessage("351914411549@c.eu", "Mensagem de texto")

    print(response.data)

if __name__ == '__main__':
    main()
