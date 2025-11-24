# Conversor de Sentenças - Lógica Proposicional

## Link para testes:
https://agent-cpc-to-79am665t4-carlos-cunhas-projects.vercel.app/

## URL do vídeo de utilização:

https://youtu.be/Z39_0DXu4co

## 📋 Escopo do Projeto

Este projeto é um conversor bidirecional entre **linguagem natural** e **cálculo proposicional clássico**, desenvolvido como parte da disciplina de Lógica para Computação da FACEF.

## 🎯 Funcionalidades Principais

1. Conversão de Linguagem Natural para Lógica
	- Recebe uma sentença em português
	- Converte para fórmula do cálculo proposicional clássico
	- Utiliza símbolos: ∧ (e), ∨ (ou), ¬ (negação), → (implica), ↔ (se e somente se)
	- Atribui letras proposicionais (p, q, r, ...)

2. Conversão de Lógica para Linguagem Natural
	- Recebe uma fórmula lógica 
	- Converte para uma sentença clara em português

3. Interface Web
	- Frontend em HTML/CSS/JavaScript
	- Alternância entre os modos de conversão
	- Exibição do resultado e mapeamento das proposições

## 🏗️ Arquitetura (resumo)

- Backend: FastAPI + integração com GenKit / Google Gemini para apoio na análise e conversão.
- Frontend: HTML/CSS/JS servidos por `index.html`.

Arquivos principais:

```
frontend/index.html       # Interface web
backend/main.py           # Servidor FastAPI com endpoints
README.MD                 # Documentação (este arquivo)
```

## 🔌 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/convert/nl-to-logic` | Converte linguagem natural para fórmula lógica |
| POST | `/convert/logic-to-nl` | Converte fórmula lógica para linguagem natural |
| GET  | `/`                    | Informações da API |

## 🚀 Como Usar Localmente (rápido)

Instalar dependências e executar o backend (ver `backend/requirements.txt`):

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py

# Em outro terminal (servir frontend):
cd ../frontend
python3 -m http.server 8080
```

Acesse `http://localhost:8080`.

## 📝 Estratégia de tradução (regras, mapeamento e uso de LLMs)



### 1) Objetivo e contrato
- Entrada: frase(s) em português ou uma fórmula proposicional simbólica.
- Saída: fórmula do cálculo proposicional com letras proposicionais (p, q, r, ...) e/ou uma sentença em português que explique o mapeamento.
- Erro/limite: O sistema pressupõe domínio proposicional (sem quantificadores) e não resolve ambiguidade pragmática sem contexto adicional.

### 2) Regras e mapeamento (NL → Lógica)

- Identificação de predicados proposicionais: frases ou orações independentes viram proposições atômicas. Ex.: "chove" → p; "a rua fica molhada" → q.
- Conectivos:
  - "e", "além disso", "mas" → ∧ (conjunção)
  - "ou" (disjunção inclusiva) → ∨
  - "não", "não é verdade que" → ¬ (negação) — atenção ao alcance da negação
  - "se... então", "caso" → → (implicação). Interpretação: condicional material por padrão
  - "se e somente se", "somente se e se" → ↔ (bicondicional)

- Precedência e parênteses: aplica-se a precedência lógica padrão (¬ > ∧ > ∨ > → > ↔) e insere parênteses explicitamente quando necessário.
- Variáveis proposicionais: atribuímos letras p, q, r... mantendo um dicionário que associa cada frase atômica à letra usada e retornamos esse mapeamento ao usuário.

### 3) Regras e mapeamento (Lógica → NL)

- Cada operador é verbalizado de forma natural: ∧ → "e", ∨ → "ou", ¬ → "não", → → "se... então...", ↔ → "se e somente se".
- Mantemos o mapeamento das letras para as descrições fornecidas (quando existirem) para produzir sentenças legíveis.

### 4) Uso da LLM (Google Gemini) e pipeline híbrido

Abordagem híbrida: LLM + regras simbólicas.

- Etapa A (pré-processamento simbólico): normalização do texto (remoção de ruído, tokenização), detecção de possíveis cláusulas atômicas usando regex/heurísticas.
- Etapa B (LLM para interpretação): a LLM é consultada com um prompt few-shot para sugerir uma estrutura lógica e possíveis mapeamentos. Usamos temperatura baixa para respostas determinísticas.
- Etapa C (verificação simbólica): parse do resultado sugerido pela LLM com um parser simples; validação de sintaxe, aplicação de precedência e consistência do mapeamento de variáveis.
- Etapa D (correção/fallback): se o output não for bem formado ou houver inconsistências, aplicam-se regras heurísticas (por exemplo, isolar cláusulas por conjunções/condicionais e reprocessar) ou re-chamar a LLM com um prompt ajustado.


## 🧪 Exemplos de input/output com análise (acertos e erros)

1) Simples — condicional direto (acerto)

- Entrada (NL): "Se chove, então a rua fica molhada"
- Saída esperada (Lógica): p → q
- Mapeamento retornado: p = "chove", q = "a rua fica molhada"

Análise: Frase direta, sujeito e consequência explícitos. O sistema normalmente acerta, pois estrutura sintática é clara e corresponde ao padrão "Se A, então B".

2) Conjunção com negação (acerto com atenção ao escopo)

- Entrada (NL): "Não chove e a rua está molhada"
- Saída possível (Lógica): ¬p ∧ q
- Mapeamento: p = "chove", q = "a rua está molhada"

Análise: O escopo da negação é "chove" — aqui o parser heurístico lida bem. Risco: em sentenças como "Não (chove e a rua está molhada)" o resultado muda; por isso, o sistema sempre tenta inserir parênteses para deixar o escopo explícito.

3) Ambiguidade de "ou" (erro comum)

- Entrada (NL): "Você pode beber chá ou café"
- Saída (Lógica) (geral): p ∨ q

Análise: Em linguagem natural, "ou" pode ser exclusivo (XOR) ou inclusivo (OR). O sistema assume disjunção inclusiva (∨). Erro: quando o domínio exige exclusividade. Mitigação: solicitar esclarecimento ao usuário ou usar heurísticas baseadas em verbos/modais.

4) Antecedente composto implícito (erro de segmentação)

- Entrada (NL): "Se chove e tem vento, a festa é cancelada"
- Saída correta: (p ∧ r) → q  (p = "chove", r = "tem vento", q = "a festa é cancelada")

Possível erro: dividir em duas implicações p → q ∧ r → q se a segmentação de cláusulas não for feita corretamente.

Mitigação: pré-processamento que detecta conjunções no antecedente e agrupa como conjunção única.

5) Frases com pronomes e implicatura (difícil para LLMs simples)

- Entrada (NL): "Se João chega atrasado, ele perde a prova"
- Saída: p → q

Desafio: resolver que "ele" refere-se a "João" — normalmente o sistema (com LLM) resolve por co-referência, mas heurísticas simples podem falhar.

6) Bicondicional natural (sucesso quando explícito)

- Entrada (NL): "Você tem ingresso se e somente se pagou"
- Saída: p ↔ q

Análise: Quando expressões como "se e somente se" aparecem explicitamente, o sistema mapeia corretamente para ↔. Problemas surgem quando a linguagem natural usa locuções indiretas.

## Fluxo da aplicação

![alt text](/assets/fluxo.png)

## Próximos passos e melhorias sugeridas

- Implementar um módulo leve de co-referência (ou usar recurso de LLM com prompt específico) para reduzir erros com pronomes.
- Fornecer uma interface de esclarecimento (quando o sistema detecta baixa confiança, perguntar ao usuário se "ou" é exclusivo, qual o escopo da negação, etc.).
- Adicionar testes unitários para o parser e para os exemplos mais comuns.

## 🛠️ Tecnologias Utilizadas

- Python 3
- FastAPI
- GenKit (camada de integração com Google Gemini)
- HTML5/CSS3/JavaScript
- Pydantic (validação de dados)

## 📚 Disciplina
Lógica para Computação - Uni-FACEF


