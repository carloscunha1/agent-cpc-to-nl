# Conversor de Sentenças - Lógica Proposicional

## 📋 Escopo do Projeto

Este projeto é um conversor bidirecional entre **linguagem natural** e **cálculo proposicional clássico**, desenvolvido como parte da disciplina de Lógica para Computação da FACEF.

## 🎯 Funcionalidades Principais

### 1. Conversão de Linguagem Natural para Lógica
- Recebe uma sentença em português
- Converte para fórmula do cálculo proposicional clássico
- Utiliza símbolos: ∧ (e), ∨ (ou), ¬ (negação), → (implica), ↔ (se e somente se)
- Atribui letras proposicionais (p, q, r, etc)

### 2. Conversão de Lógica para Linguagem Natural
- Recebe uma fórmula lógica em notação simbólica
- Converta para uma sentença clara em português
- Facilita a compreensão da expressão lógica

### 3. Interface Web Intuitiva
- Frontend em HTML/CSS/JavaScript
- Alternância entre dois modos de conversão
- Exibição de resultado com explicação

## 🏗️ Arquitetura

### Backend
- **FastAPI**: Framework web para API REST
- **Genkit**: Framework de IA generativa
- **Google Gemini 2.5 Flash**: Modelo de linguagem para conversões
- **CORS**: Habilitado para requisições cross-origin

### Frontend
- HTML5 com interface responsiva
- JavaScript vanilla para comunicação com API
- Suporte a atalho Ctrl+Enter para conversão rápida

## 📁 Estrutura de Arquivos

```
.
├── index.html              # Interface web
├── backend/
│   └── main.py            # Servidor FastAPI com endpoints
├── .gitignore             # Configurações do Git
├── .env                   # Variáveis de ambiente (não versionado)
└── README.MD              # Este arquivo
```

## 🔌 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/convert/nl-to-logic` | Converte linguagem natural para fórmula lógica |
| POST | `/convert/logic-to-nl` | Converte fórmula lógica para linguagem natural |
| GET | `/` | Informações da API |

## 🚀 Como Usar

### Pré-requisitos
- Python 3.8+
- Chave de API do Google Gemini

### Instalação do Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuração
Crie um arquivo `.env` na raiz do projeto:
```
GOOGLE_GENAI_API_KEY=sua_chave_aqui
```

### Executar
```bash
# Terminal 1 - Backend
python backend/main.py

# Terminal 2 - Frontend (servir em localhost)
python -m http.server 8080
```

Acesse `http://localhost:8080` no navegador.

## 📝 Exemplo de Uso

**Entrada (Linguagem Natural):**
> Se chove, então a rua fica molhada

**Saída (Lógica Proposicional):**
> p → q (onde p = "chove" e q = "rua fica molhada")

## 🛠️ Tecnologias Utilizadas

- Python 3
- FastAPI
- Google Genkit
- Google Gemini AI
- HTML5/CSS3/JavaScript
- Pydantic (validação de dados)

## 📚 Disciplina
Lógica para Computação - FACEF
