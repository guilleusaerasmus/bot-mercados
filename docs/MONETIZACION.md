# Plan de monetización

Resumen honesto: el bot ya trae toda la **infraestructura para cobrar** (planes, suscripción con
Telegram Stars, invitaciones, métricas), pero **antes de cobrar a desconocidos hay que cambiar la
fuente de datos** por una con licencia comercial y revisar los aspectos legales (secciones 4 y 5).
Lo más rentable a corto plazo no es la suscripción, sino usar el bot como escaparate para vender
bots a medida (sección 6).

## 1. Modelo freemium (ya implementado)

| | Gratis | Premium (`PREMIUM_PRICE_STARS`, por defecto 250 ⭐ / 30 días) |
|---|---|---|
| Preguntas a la IA (escritas o por voz) | 10 al día | 60 al día |
| Alertas de precio | 3 | 30 |
| Lista de seguimiento | 5 valores | 50 valores |
| Cartera real (con FIFO e IRPF) | 5 valores | 50 valores |
| PDFs para preguntarle (RAG) | 3 | 30 |
| Resumen diario automático (lista + cartera + resultados próximos) | — | ✅ laborables a las 18:00 |
| Comandos rápidos (/cotizacion, /grafico, /ficha, /top, /cartera, /mercado, /quiz) | ilimitados | ilimitados |

La idea: lo que **no cuesta dinero** (comandos sin IA) es ilimitado para todos y engancha; lo que
**cuesta tokens** (la IA) y lo que da **comodidad** (alertas, resumen diario) es lo que se paga.

Cómo funciona el cobro, ya programado en `telegram_bot.py`:
- `/premium` crea una factura de suscripción en Stars (`XTR`, 30 días; Telegram solo admite ese periodo).
- El pre-checkout comprueba factura e importe (Telegram exige responder en menos de 10 s).
- El pago activa Premium hasta la fecha que da Telegram, sin pisar días regalados. Las renovaciones
  llegan como nuevos pagos y los avisos duplicados no cuentan dos veces.
- `/cancelar` cancela la renovación y mantiene Premium hasta el final del periodo pagado.
- `/terms` y `/paysupport` son **obligatorios** para vender productos digitales en Telegram.
- Para el administrador: `/stats` (métricas), `/regalar ID DÍAS` y `/reembolsar CHARGE_ID`.

**Por qué Stars:** Telegram obliga a cobrar los productos digitales en Stars (no se permiten otras
pasarelas). Los Stars se retiran a través de Fragment, cobrando en TON. Según fuentes secundarias
(no oficiales), se pueden retirar 21 días después de ganarlos, a partir de 1.000 ⭐, y el
desarrollador recibe aproximadamente 0,013 $ por Star. Con esas cifras, 250 ⭐ ≈ **3,25 $ netos por
suscriptor y mes**. Verifica el valor actual en tu panel de Telegram antes de fijar el precio.

## 2. Costes reales de la IA

Medido en el código: cada llamada al modelo lleva unos **2.400 tokens fijos** (instrucciones más las
16 herramientas). Una pregunta típica hace 2-3 llamadas con resultados intermedios, así que gasta
**7.000-10.000 tokens de entrada y unos 500 de salida**. Las preguntas sobre documentos añaden
~1.000 tokens (6 fragmentos). Los embeddings del RAG se calculan en local y no cuestan nada.

**Plan gratuito de Groq** (límites orientativos por modelo; los exactos están en tu consola, en
*Settings → Limits*): unos 8.000 tokens por minuto y 200.000 al día. Con los 3 modelos de reserva
que usa el bot, sale a **unas 60-70 preguntas a la IA al día en total**. Basta para pruebas y para
los primeros 10-30 usuarios activos, no para más.

**Plan de pago de Groq (Developer)** con `gpt-oss-120b`: unos 0,15 $ por millón de tokens de
entrada y 0,60 $ por millón de salida (precio publicado por terceros en octubre de 2026; confírmalo
en la consola). Eso son **unos 0,0017 $ por pregunta** (0,16 céntimos). Las notas de voz usan
Whisper, que tiene su propio cupo gratuito de audio.

| Escenario por usuario y mes | Preguntas | Coste IA |
|---|---|---|
| Gratis típico (2 al día) | 60 | 0,10 $ |
| Gratis al máximo (10 al día) | 300 | 0,51 $ |
| Premium típico (5 al día) | 150 | 0,26 $ |
| Premium al máximo (60 al día) | 1.800 | 3,06 $ (justo por debajo de los ~3,25 $ que ingresa) |

Palancas para mejorar el margen si hace falta:
- Bajar el cupo gratuito.
- Subir el precio.
- Servir a los gratuitos con el modelo pequeño (`gpt-oss-20b`).
- Aprovechar el descuento de Groq en tokens repetidos: las instrucciones y las herramientas son
  siempre iguales.

## 3. Cuentas de ejemplo

Con una conversión típica de freemium del 2-5 % (hipótesis, no un dato del bot), por cada 1.000
usuarios activos al mes:

- **Ingresos:** 20-50 suscriptores × 3,25 $ ≈ **65-160 $ al mes**.
- **Coste de IA:** unos 1.000 × 0,10 $ ≈ **100 $ al mes** en el peor caso, si todos fuesen activos
  diarios (en la práctica, muchos menos).
- **Datos con licencia:** desde unos 20-100 $ al mes en planes publicados (por ejemplo, EODHD). El
  uso comercial y la redistribución se presupuestan aparte, así que hay que pedir precio.
- **Servidor:** 0 € (sección 7).

Conclusión: es un **proyecto paralelo que puede pagarse solo y dejar algo de margen** con unos
cientos de usuarios fieles, no un negocio grande por sí mismo. Su gran valor está en el CV y como
escaparate para la sección 6.

## 4. Lo que hay que cambiar antes de cobrar

1. **Datos con licencia comercial.** Las condiciones de yfinance (Yahoo) y del RSS de Google News
   solo permiten uso personal y no comercial.
   - Todo el acceso a datos está aislado en `bot_mercados/mercado.py`, así que migrar significa
     reescribir solo ese archivo.
   - Candidatos: EODHD, Financial Modeling Prep, Twelve Data. Pide por escrito qué permite cada plan
     (mostrar datos a tus usuarios = redistribución).
   - Las bolsas imponen además sus propias condiciones a los datos en tiempo real; los datos de
     cierre o con retraso suelen ser más baratos.
2. **Servidor 24/7.** Las alertas y los resúmenes solo funcionan si el bot está encendido (sección 7).
3. **Copias de seguridad** de la base de datos, que contiene los pagos (sección 7).
4. **Política de privacidad completa.** `/terms` ya resume qué se guarda. Falta añadir quién es el
   responsable y que las preguntas se procesan en Groq (EE. UU.).

## 5. Aspectos legales en España (orientativo, no es asesoramiento jurídico)

- **Información sí, recomendaciones personalizadas no.** El asesoramiento en inversiones requiere
  autorización de la CNMV. El bot está diseñado para informar y formar: las instrucciones de la IA
  le prohíben recomendar comprar o vender, y todos los mensajes llevan el aviso. Mantenlo así.
- **Recomendaciones públicas.** La normativa europea de abuso de mercado (Reglamento 596/2014)
  impone obligaciones de presentación y de declarar conflictos de interés a quien difunde
  recomendaciones de inversión. Otra razón para que el bot no las haga.
- **Publicidad de brokers (afiliados).** La publicidad de servicios de inversión está regulada por
  la CNMV, también cuando la hacen influencers. Si algún día metes enlaces de afiliado, consulta
  antes las normas vigentes.
- **Hacienda.** Los ingresos por Stars tributan. Con cobros recurrentes probablemente tendrás que
  darte de alta (modelo 036/037) y valorar si te corresponde ser autónomo. Consúltalo con una gestoría.
- **Protección de datos (RGPD).** Guarda lo mínimo (el bot ya lo hace), informa en `/terms` y
  atiende las peticiones de borrado.

## 6. Otras vías de ingresos, de más a menos rentable

1. **Bots a medida para negocios (B2B).** Es la misma arquitectura (agente + herramientas + pagos)
   con otras herramientas: reservas, preguntas frecuentes, catálogo o seguimiento de pedidos.
   - Se cobra por montarlo y una cuota mensual de mantenimiento.
   - Sin riesgo de licencias de datos ni de CNMV. Este bot es tu demostración.
2. **Marca blanca para el sector financiero educativo:** clubs de inversión universitarios,
   academias de finanzas, newsletters y asesores con licencia (que sí pueden recomendar bajo su
   autorización).
   - Empieza por un **piloto con el club de inversión o algún profesor de CUNEF**: primer cliente y
     carta de recomendación para el CV.
3. **Informes de pago puntuales con Stars,** por ejemplo un informe mensual del IBEX o una revisión
   de cartera en PDF. Telegram admite pagos únicos además de suscripciones.
4. **Patrocinio del canal** cuando tenga audiencia. Si el patrocinador vende productos financieros,
   aplica lo dicho sobre publicidad.

## 7. Puesta en producción (coste 0)

- **Servidor:** Oracle Cloud *Always Free* (máquinas ARM).
  - Algunas fuentes de 2026 hablan de 4 núcleos y 24 GB; otras, de un recorte a 2 núcleos y 12 GB
    para cuentas nuevas. Cualquiera de las dos sobra para este bot.
  - Otra opción: dejarlo en tu PC encendido mientras pruebas.
- **Servicio en Linux** (Ubuntu; no probado en este entorno):
  ```bash
  sudo apt install -y python3-venv git
  git clone <tu-repo> ~/bot-mercados && cd ~/bot-mercados
  python3 -m venv ~/.venvs/bot-mercados && ~/.venvs/bot-mercados/bin/pip install -r requirements.txt
  cp .env.example .env && nano .env
  sudo tee /etc/systemd/system/bot-mercados.service >/dev/null <<'EOF'
  [Unit]
  Description=Bot de Mercados
  After=network-online.target
  [Service]
  User=ubuntu
  WorkingDirectory=/home/ubuntu/bot-mercados
  ExecStart=/home/ubuntu/.venvs/bot-mercados/bin/python -m bot_mercados telegram
  Restart=always
  [Install]
  WantedBy=multi-user.target
  EOF
  sudo systemctl enable --now bot-mercados && journalctl -u bot-mercados -f
  ```
- **Copia de seguridad diaria** de la base de datos (incluye los pagos): una tarea de cron con
  `sqlite3 ~/.bot-mercados/bot.db ".backup /ruta/copia-$(date +%F).db"`.

## 8. Plan de lanzamiento

| Fase | Qué hacer | Señal para pasar a la siguiente |
|---|---|---|
| 0. Prueba privada (ya) | `ALLOWED_USER_IDS` con 5-10 amigos. Revisa `/stats` y pide opiniones | Usan el bot varios días seguidos sin que se lo recuerdes |
| 1. Beta pública gratuita | Canal público con `CHANNEL_ID`, `/invitar`, comparte resultados del simulador en redes. Premium regalado con `/regalar` a los primeros | Más de 100 usuarios activos al mes y retención a 7 días por encima del 20 % |
| 2. Cobro | Datos con licencia, servidor 24/7, privacidad completa y gestoría. Activa Premium | Primeros 10 suscriptores. Mide la conversión y las cancelaciones |
| 3. B2B | Ofrece la versión marca blanca y bots a medida usando tus métricas como prueba | Primer cliente |

**Idea de contenido para crecer:** «¿Y si hubieras invertido 100 € al mes en X desde hace 10 años?»
con el gráfico del bot. Es contenido corto, visual y compartible (TikTok, Instagram, X) que lleva
directamente a probar el simulador.

## Fuentes

- Telegram, pagos de bienes digitales con Stars (obligatoriedad, `/paysupport`, `/terms`, plazo del pre-checkout): <https://core.telegram.org/bots/payments-stars>
- Telegram, suscripciones de 30 días: <https://core.telegram.org/api/subscriptions>
- Retirada de Stars y valor estimado (fuentes secundarias): <https://postmypost.io/resources/telegram-stars-what-are-they-how-much-do-the-stars-cost-and-how-to-withdraw-them>, <https://www.such.chat/blog/withdraw-telegram-stars-and-convert-them-to-ton>
- Groq, límites: <https://console.groq.com/docs/rate-limits> · precios: <https://www.morphllm.com/groq-pricing>
- Precios de EODHD: <https://eodhd.com/pricing> · comparativa de APIs de datos: <https://multiples.vc/compare/api-pricing>
- Oracle Always Free en 2026: <https://braindetox.kr/en/posts/oracle_always_free_tier_reduced_2026.html>
- Escala del ahorro del IRPF 2026: <https://guiafiscal.es/irpf/tramos-irpf-2026/>
