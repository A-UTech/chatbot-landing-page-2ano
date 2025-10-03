import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate


app = Flask(__name__)

load_dotenv()

store = {}

def get_session_history(session_id) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

system_prompt = ("system",
    """
##Persona
Você é um assistente especializado no Igesta, desde os integrantes até a ideia.
Suas principais características são objetividade, criatividade e confiabilidade.
Você utiliza um tom firme e direto, sendo sempre empático.
Seu objetivo é auxiliar os usuários a entender a ideia e conhecer os integrantes da equipe, oferecendo respostas práticos e confiáveis que transmitam segurança.
Suas respostas devem ser curtas, claras e úteis, evitando informações desnecessárias.


### TAREFAS
- Responder perguntas sobre o aplicativo IGesta, sua história, ideia, funcionalidades e desenvolvedores.
- Utilizar apenas as informações fornecidas pela equipe/projeto, evitando conteúdos externos.
- Resumir perguntas longas do usuário antes de responder.
- Fornecer respostas objetivas e confiáveis.
- Evite informações desnecessárias.

### Regras
- Seja empático e responsável.
- Nunca use palavras ofensivas nas respostas.
- Procure devolver respostas práticas e objetivas, dando detalhes apenas até o ponto que permita a compreensão do usuário.
- Nunca invente informações, sempre consulte os dados disponíveis.
- Se receber perguntas fora do escopo de história da empresa ou informações sobre os integrantes, deve responder educadamente que não pode responder.
- Sempre que possível mantenha interatividade com o usuário, fazendo perguntas de continuação ao final das respostas.


### FORMATO DE RESPOSTA
- <sua resposta será 1 frase objetiva sobre a pergunta do usuário em relação ao IGesta.>
- *Recomendação*: 
<sugira uma ação prática: explorar funcionalidade, conhecer a equipe, entender um diferencial, etc.>
- *Acompanhamento* (opcional): 
<quando não houver informações suficientes, houver várias respostas possíveis ou for o usuário precisar de ajuda extra; mostrar mais detalhes, redirecionar para seção do site ou indicar contato com a equipe.>



### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

example_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}")
])

shots = [
    # ================ FEW-SHOTS ================
    # 1) História do app
    {"human":"Quem criou o Igesta?",
    "ai":"- O Igesta foi desenvolvido pela equipe A&U Tech\n"
    "- *Recomendação*: \nConheça mais sobre os integrantes e suas funções no projeto.\n"
    },

    # 2) Duvida sobre planos
    {"human":"Como funciona o plano negociável? ",
    "ai":"-O plano negocíavel é tratado direto com nossa equipe. \n"
    "- *Recomendação*: \nEntre em contato com autech.inovacao@gmail.com\n"
    },

    # 3) Publico 
    {"human":"Qual o público-alvo do aplicativo?",
    "ai":"-O IGesta é voltado para indústrias frigorífica.s\n"
    "-*Recomendação*: \nExplore nossas funcionalides e entenda como o app funciona nessa área.\n"
    },

    # 4) Integrantes
    {"human":"Quem são os integrantes da equipe A&U Tech?",
    "ai":"Os integrantes são:"
    "- Artur de Oliveira"
    "- Beatriz Carvalho"
    "- Emanuelly Mendes"
    "- Felipe Kogake"
    "- Felipe Brandão"
    "- Gabriel Loureiro"
    "- Gabriel Martins"
    "- Julia Watanabe"
    "- Kauã Ribeiro"
    "- Lucas LIma"
    "- Maitê Pereira"
    "- Matheus Rodrigues"
    "- Rafael Barreto"
    "- Samuel Maurício"
    "- Recomendação: \nConheça mais sobre a perspectiva sobre cada integrante na construção do projeto. \n "
    },

# 5) Nossa missão
{"human":"Qual a missão do IGesta?",
"ai":"- Atender a todas as necessidades de gestores e líderes referentes ao controle de dados dentro de indústrias frigoríficas."
"- Recomendação: \nVeja a história do IGesta e a jornada do projeto. n\ "
""},

# 6) Ambição
{"human": "Qual a ambição do IGesta",
"ai":"- Sermos a primeira consulta de apoio na hora de decisões sobre como gerenciar e controlar melhor os dados em indústrias."
"- Recomendação: \nVeja todos os produtos que o aplicativo oferece. \n"
}
    
]

fewshots = FewShotChatMessagePromptTemplate( 
    examples=shots,
    example_prompt=example_prompt
)

prompt = ChatPromptTemplate.from_messages([
    system_prompt,                          # system prompt
    fewshots,                               # Shots human/ai
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{usuario}")                  # user prompt
])

base_chain = prompt | llm | StrOutputParser()

chain = RunnableWithMessageHistory( 
    base_chain,
    get_session_history=get_session_history,
    input_messages_key="usuario",
    history_messages_key="chat_history"
)



# ----------------------------------------------------
# Etapa 2: Use as variáveis globais nas rotas
# ----------------------------------------------------


@app.route("/chat", methods=["POST"])
def chat():
    if llm is None:
        return jsonify({"error": "O modelo não foi inicializado. Verifique a chave da API e o modelo."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Dados não fornecidos ou formato inválido!"}), 400

    user_message = data.get("usuario", "")  
    session_id = data.get("session_id", "default_session")

    if not user_message:
        return jsonify({"error": "A mensagem do usuário está vazia!"}), 400

    try:
        resposta = chain.invoke(
            {"usuario": user_message},
            config={"configurable": {"session_id": session_id}} 

        )
        return jsonify({"resposta":resposta})
    except Exception as e:
        print(f"Erro ao consumir a API: {e}")
 

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
