# 📈 Bot de Mercados — analista de bolsa con IA en Telegram

Bot de Telegram que responde preguntas de bolsa en lenguaje natural con **datos reales**, lee
**informes anuales y PDFs** (RAG con búsqueda híbrida y citas), lleva tu **cartera** con fiscalidad
española, genera gráficos, vigila precios y enseña finanzas. Por dentro es un **agente LLM con
tool calling**: el modelo decide qué herramienta necesita, el programa la ejecuta y el modelo
explica los resultados sin inventar cifras.

Incluye un **modelo de negocio completo**: plan gratuito y Premium por suscripción con Telegram
Stars, invitaciones con recompensa, canal público automático y métricas para el administrador.

**Coste de arranque: 0 €.** Groq (IA y voz, plan gratuito), embeddings locales, Telegram y datos públicos.

## Qué sabe hacer

| Pregúntale... | Qué hace por dentro |
|---|---|
| «¿Qué opinas de Inditex?» | `ficha_empresa`: valoración (PER, EV/EBITDA, PEG), márgenes, ROE, deuda neta/EBITDA, crecimiento, dividendo de los últimos 12 meses y payout, consenso y precio objetivo de analistas, señales técnicas (medias 50/200, cruces, RSI, MACD) y próximos resultados |
| «¿Qué riesgos destaca Apple en su informe anual?» | `buscar_en_documentos` (**RAG**): descarga e indexa el 10-K oficial de la SEC y responde citando el apartado («Item 1A. Risk Factors») |
| *(envías un PDF)* «¿Qué dividendo propone?» | **RAG** sobre tus documentos: responde citando la página |
| «Compara Santander, BBVA y el IBEX en 5 años con un gráfico» | `rendimiento` (rentabilidad anualizada, volatilidad, Sharpe, máxima caída, en euros si mezclas monedas) + `grafico` en base 100 |
| «¿Qué empresas españolas pagan más dividendo?» | `buscar_acciones`: buscador en vivo de la bolsa española o de EE. UU. por dividendo, PER, subidas o bajadas y sector |
| «He comprado 100 Santander a 9,50» · «¿Cómo va mi cartera?» | `mi_cartera`: compras y ventas por lotes, ganancias en euros, ventas por **FIFO** (como exige Hacienda), aviso de la **regla de los dos meses** e IRPF estimado |
| «Si hubiera invertido 200 € al mes en el S&P 500 desde hace 10 años...» | `simular_inversion`: dividendos reinvertidos, en euros, TIR (XIRR) y **gráfico para compartir** |
| «Analiza mi cartera: 40 % SAN.MC, 30 % AAPL, 30 % IBE.MC» | `analizar_cartera`: métricas en euros, correlaciones, riesgo que elimina la diversificación, concentración por divisa y sector |
| «Gané 8.000 € vendiendo acciones, ¿cuánto pago?» | `calcular_impuestos`: IRPF 2026 de la base del ahorro (19-30 %) con compensación de pérdidas |
| «¿Qué pasa hoy en los mercados?» · «¿Por qué cae Iberdrola?» | `resumen_mercado`, `noticias` |
| «Avísame si Santander baja de 10 €» | `alertas`: el bot revisa cada 10 minutos y te escribe |
| 🎙️ *(nota de voz)* | Transcripción con Whisper (Groq) y respuesta como a un mensaje escrito |

**Comandos rápidos sin IA** (ilimitados): `/mercado`, `/cotizacion`, `/ficha`, `/grafico`, `/top`,
`/cartera`, `/compra`, `/venta`, `/documentos`, `/noticias`, `/lista`, `/alertas`, `/quiz`.
**Modo inline:** escribe `@tu_bot SAN.MC` en cualquier chat para compartir una cotización.

## RAG: preguntas sobre informes y documentos

```
PDF del usuario ─┐                         ┌─► BM25 (palabras clave, lematizado es/en) ─┐
                 ├─► trocear (~550 car.,    │                                            ├─► RRF ─► 6 fragmentos ─► LLM cita [1][2]
10-K de la SEC ──┘   por frases y páginas) ─┴─► embeddings locales (MiniLM multilingüe) ┘
                     └─► SQLite (texto + vector + «pág. 12» / «Item 1A»)
```

- **Embeddings locales** con `fastembed` (ONNX, sin PyTorch ni API de pago):
  `paraphrase-multilingual-MiniLM-L12-v2`, que **funciona entre idiomas** (preguntas en español
  sobre un 10-K en inglés).
- **Búsqueda híbrida.** La similitud de vectores entiende paráfrasis; BM25 acierta nombres
  propios, cifras y siglas. Las dos listas se combinan con *Reciprocal Rank Fusion*.
- **Troceado consciente del modelo.** MiniLM solo lee ~128 tokens, así que los fragmentos son
  de ~550 caracteres, cortados por frases con solapamiento y guardando la página o el apartado.
- **Citas verificables.** Cada fragmento lleva su documento y su ubicación, y el agente debe
  citarlos y reconocer cuando la respuesta no está en los documentos.
- **Privacidad.** Los PDFs de cada usuario solo los ve él. Los 10-K son públicos y se comparten
  y se indexan una sola vez.

**Evaluación** (`python -m bot_mercados evaluar-rag`): 20 preguntas sobre un informe de prueba
(literales, paráfrasis sin palabras en común, en inglés y con términos exactos):

| Modo | Acierto a la primera (hit@1) | Entre los 3 primeros (hit@3) | MRR |
|---|---|---|---|
| BM25 (palabras clave) | 70 % | 80 % | 0,75 |
| Vectores | 100 % | 100 % | 1,00 |
| **Híbrido (por defecto)** | **95 %** | **100 %** | **0,97** |

Lo que enseña la evaluación:
- Lematizar BM25 subió su acierto del 55 % al 70 %, y el del híbrido del 85 % al 95 %.
- En un documento corto con temas muy distintos los vectores solos ganan.
- En el 10-K real de Apple (521 fragmentos), el híbrido encontró el párrafo sobre dependencia
  de fabricantes en Asia que los vectores solos no recuperaban. Por eso es el modo por defecto
  (configurable con `RAG_EMBEDDINGS`).
- Limitación conocida: la calidad depende de cómo se formule la pregunta. Una mejora futura es
  reescribirla o lanzar varias consultas (*multi-query*).

## Arquitectura

```
Telegram ─► telegram_bot.py ─► planes.py (límites, Premium, invitaciones) ─► db.py (SQLite)
                │
                ├─ comandos rápidos ─► tools.py ──────────────────────────────┐
                ├─ texto y voz (voz.py: Whisper) ─► agent.py ◄─► LLM (Groq / Grok)
                │                                    │ tool_calls              ▼
                └─ PDFs ─► documentos.py ─► rag.py   └──► tools.py ─► analisis.py (finanzas puras)
                                (pypdf, SEC)  (híbrido)                ─► graficos.py (matplotlib)
                                                                       ─► mercado.py (Yahoo, BCE, Google News)
Tareas programadas: alertas cada 10 min · resumen diario Premium · post diario en el canal
```

| Módulo | Responsabilidad |
|---|---|
| `agent.py` | Bucle de tool calling, historial por chat, reserva de modelos si uno agota su cupo gratis |
| `tools.py` | 16 herramientas con esquemas JSON compactos (un test vigila su tamaño) y validación de argumentos |
| `rag.py` · `documentos.py` · `rag_eval.py` | Troceado, BM25, embeddings, RRF, PDFs, SEC EDGAR y evaluación |
| `analisis.py` | Finanzas puras y testeables: RSI, MACD, drawdown, Sharpe, cartera, XIRR, FIFO, IRPF |
| `mercado.py` | Única puerta a los datos de mercado, con caché: cambiar de proveedor = tocar un archivo |
| `planes.py` · `db.py` | Reglas de negocio y persistencia con migraciones: usuarios, uso, alertas, lotes, ventas, pagos |
| `telegram_bot.py` | Comandos, botones, inline, voz, PDFs, pagos con Stars, quiz, anuncios y tareas programadas |

Decisiones de diseño:
- **Tokens como recurso escaso.** Esquemas compactos, campos vacíos eliminados de las respuestas,
  historial sin pasos intermedios y `reasoning_effort=low`.
- **Lo determinista no pasa por la IA.** Precios, gráficos, fichas, buscador y cartera por
  comando cuestan 0 tokens.
- **Datos que no engañan.** La rentabilidad por dividendo de Yahoo era incoherente (1,1 % para
  Inditex cuando pagó un 3 %), así que se calcula con lo realmente cobrado en 12 meses. El
  buscador reordena por el dato que muestra y marca dividendos anómalos.
- **Pagos y ventas robustos.** El pre-checkout valida la factura; los pagos duplicados no cuentan
  dos veces; las ventas FIFO son transaccionales; las preguntas que fallan no gastan cupo.
- **Seguridad.** Claves en `.env`, argumentos del modelo validados, SQL parametrizado, cada usuario
  solo accede a sus alertas, cartera y documentos, y hay freno anti-abuso.

## Puesta en marcha

1. **Clave de IA (gratis):** <https://console.groq.com> → *API Keys*.
2. **Bot:** [@BotFather](https://t.me/BotFather) → `/newbot` (y `/setinline` para activar el modo inline).
3. **Instalación** (Python 3.10+):
   ```powershell
   python -m venv $HOME\.venvs\bot-mercados
   & $HOME\.venvs\bot-mercados\Scripts\Activate.ps1
   pip install -r requirements.txt
   copy .env.example .env   # rellena LLM_API_KEY, TELEGRAM_TOKEN, ADMIN_USER_IDS y SEC_USER_AGENT
   ```
4. **Ejecución:**
   ```powershell
   python -m bot_mercados consola -v          # prueba en la terminal; -v muestra las herramientas que usa
   python -m bot_mercados indexar informe.pdf # añade PDFs para preguntarle en la consola
   python -m bot_mercados evaluar-rag         # mide la búsqueda en documentos
   python -m bot_mercados telegram            # arranca el bot
   ```
5. **Tests** (121, sin internet ni cuota; también en GitHub Actions): `python -m pytest`

La primera vez que se usa el RAG se descarga el modelo de embeddings (~220 MB) en
`~/.bot-mercados/modelos`.

## Despliegue 24/7

En un servidor Ubuntu (por ejemplo, Oracle Cloud *Always Free*):
```bash
curl -fsSL https://raw.githubusercontent.com/guilleusaerasmus/bot-mercados/main/deploy/instalar.sh -o instalar.sh
bash instalar.sh        # instala, crea el servicio systemd y la copia de seguridad diaria
# copia tu .env a ~/bot-mercados/.env y: sudo systemctl restart bot-mercados
```
El bot solo usa conexiones salientes (no hay que abrir puertos). Solo puede haber **una**
instancia conectada a Telegram a la vez: para el bot del PC antes de arrancar el del servidor.

## Monetización

Ver [docs/MONETIZACION.md](docs/MONETIZACION.md): modelo freemium con Telegram Stars, costes por
pregunta, capacidad del plan gratuito, qué cambiar antes de cobrar y plan de lanzamiento.

## Limitaciones

- **Licencias:** yfinance (Yahoo) y el RSS de Google News solo permiten uso personal. Sirven para el
  prototipo y el portafolio; para cobrar hay que pasar `mercado.py` a un proveedor con licencia.
  Los informes de la SEC sí son públicos.
- Datos con hasta ~15 minutos de retraso. El historial de conversación vive en memoria.
- Los informes de la SEC solo cubren empresas que cotizan en EE. UU.; para las españolas, PDF.
- Es información y formación, no asesoramiento financiero.

## Tecnologías

Python · agentes LLM con tool calling · **RAG** (búsqueda híbrida: embeddings + BM25 + RRF,
evaluada con hit@k y MRR) · fastembed/ONNX · API compatible con OpenAI (Groq/xAI) · Whisper ·
python-telegram-bot (async, JobQueue, inline, pagos con Stars) · pandas · matplotlib · SQLite ·
SEC EDGAR · pytest · GitHub Actions
