# Chatbot - IGesta


## Descrição
O chatbot que auxilia os usuários do site da A&U Tech a esclarecer dúvidas sobre a empresa, seus funcionários e a missão.

 ### Qual problema ele resolve?

&nbsp; O chatbot atua no sentido de dúvida dos usuários. Qualquer dúvida a respeito da empresa e das informações do site, o chat pode ajudar a esclarecer.

## Funcionalidades
- 💭 Dúvidas do projeto IGesta
- 💡 Dúvidas da empresa A&U Tech
- 📊 Informações sobre funcionários

## Tecnologias Utilizadas
- Flask
- Python
- Langchain
- Redis

## Exemplo de requisição e resposta
### Requisição
```
POST https://chatbot-landing-page-2ano.vercel.app/chat
Content-Type: application/json

{
  "usuario": "Olá, o que é a A&U Tech?",
  "session_id": "<numero-gerado-js>"
}
```
### Resposta
```
{
  "resposta": "A&U Tech é a equipe responsável pelo desenvolvimento do Igesta.\n- *Recomendação*:\nConheça os integrantes da A&U Tech e suas contribuições para o projeto.",
  "session_id": "<numero-gerado-js>"
}
```
