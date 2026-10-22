"""Preguntas tipo test de finanzas para /quiz (encuestas de Telegram en modo quiz).

Están escritas a mano (no las genera la IA): así son siempre correctas y no gastan tokens.
Límites de Telegram: pregunta <= 300 caracteres, opciones <= 100 y explicación <= 200.
"""

import random

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    text: str
    options: tuple[str, ...]
    correct: int
    explanation: str


_RNG = random.Random()

QUESTIONS = [
    Question("¿Qué mide el PER de una empresa?",
             ("Cuántos años de beneficios actuales pagas al comprarla", "Su nivel de deuda", "Su rentabilidad por dividendo", "Su volatilidad"),
             0, "PER = precio / beneficio por acción. Un PER de 15 significa que pagas 15 veces el beneficio anual."),
    Question("Una acción con beta 1,5 respecto a su índice...",
             ("Suele moverse un 50 % más que el índice", "Rinde un 1,5 % al año", "Tiene la mitad de riesgo que el índice", "Paga 1,5 € de dividendo"),
             0, "La beta mide la sensibilidad al mercado: con beta 1,5, si el índice sube un 10 %, la acción tiende a subir un 15 %."),
    Question("Si una inversión cae un 50 %, ¿cuánto tiene que subir después para recuperar lo perdido?",
             ("50 %", "75 %", "100 %", "150 %"),
             2, "De 100 a 50 hay que duplicar: subir un 100 %. Por eso controlar las caídas importa tanto."),
    Question("¿Qué es la rentabilidad por dividendo?",
             ("Dividendo anual por acción / precio de la acción", "Beneficio / ventas", "Dividendo / beneficio", "Subida del precio en un año"),
             0, "Si una acción vale 20 € y paga 1 € al año, su rentabilidad por dividendo es del 5 %."),
    Question("El 'payout' de una empresa es...",
             ("El porcentaje del beneficio que reparte como dividendo", "Su deuda neta", "El sueldo del consejo", "El precio objetivo de los analistas"),
             0, "Un payout muy alto (cerca o por encima del 100 %) puede indicar que el dividendo no es sostenible."),
    Question("¿Qué es un ETF?",
             ("Un fondo que cotiza en bolsa como una acción", "Un tipo de bono del Estado", "Un derivado apalancado siempre", "Una cuenta remunerada"),
             0, "Los ETF suelen replicar un índice (S&P 500, MSCI World...) con comisiones bajas y se compran y venden en bolsa."),
    Question("¿Qué mide la volatilidad?",
             ("Cuánto oscila el precio de un activo", "Cuánto dividendo paga", "Su capitalización", "Su número de accionistas"),
             0, "Se suele medir como la desviación típica anualizada de las rentabilidades diarias. Más volatilidad = más incertidumbre."),
    Question("En España (2026), ¿a qué tipo tributan los primeros 6.000 € de ganancias por vender acciones?",
             ("19 %", "15 %", "21 %", "Según tu tramo de nómina"),
             0, "La base del ahorro tiene su propia escala: 19 % hasta 6.000 €, 21 % hasta 50.000 €, 23 %, 27 % y 30 %."),
    Question("¿Qué es la diversificación?",
             ("Repartir el dinero en activos que no se mueven igual", "Comprar muchas acciones de la misma empresa", "Invertir todo en el mejor fondo", "Vender cuando el mercado cae"),
             0, "Combinar activos poco correlacionados reduce el riesgo total sin renunciar necesariamente a rentabilidad."),
    Question("Invertir la misma cantidad cada mes, pase lo que pase con el precio, se llama...",
             ("Dollar cost averaging (aportaciones periódicas)", "Market timing", "Trading intradía", "Apalancamiento"),
             0, "Al aportar siempre lo mismo compras más participaciones cuando el precio está bajo y menos cuando está alto."),
    Question("¿Qué indica un RSI por encima de 70?",
             ("Posible sobrecompra tras fuertes subidas", "Que la empresa está en quiebra", "Un dividendo alto", "Que la acción es barata"),
             0, "El RSI va de 0 a 100. Por encima de 70 se habla de sobrecompra y por debajo de 30 de sobreventa. No es una señal infalible."),
    Question("Un 'cruce dorado' ocurre cuando...",
             ("La media de 50 sesiones cruza al alza la de 200", "El oro sube más que la bolsa", "Una empresa sale a bolsa", "El PER baja de 10"),
             0, "Se considera una señal técnica alcista. Lo contrario (50 cruza a la baja la de 200) es el 'cruce de la muerte'."),
    Question("¿Qué es la capitalización bursátil?",
             ("Precio de la acción × número de acciones", "Beneficio anual", "Dinero en caja", "Ventas anuales"),
             0, "Es lo que el mercado dice que vale toda la empresa en bolsa."),
    Question("Si los tipos de interés suben, ¿qué suele pasar con el precio de los bonos que ya existen?",
             ("Baja", "Sube", "No cambia", "Se duplica"),
             0, "Los bonos nuevos pagan más, así que los antiguos (con cupón menor) valen menos. Precio y tipo van en sentido contrario."),
    Question("¿Qué significa que una acción cotice con sufijo .MC en Yahoo Finance?",
             ("Que cotiza en la Bolsa de Madrid", "Que es de Mónaco", "Que es un ETF", "Que paga dividendo mensual"),
             0, "Por ejemplo SAN.MC (Santander) o ITX.MC (Inditex). Las de EE. UU. no llevan sufijo (AAPL, MSFT)."),
    Question("El ROE (return on equity) mide...",
             ("El beneficio que genera la empresa con el dinero de sus accionistas", "La rentabilidad de sus bonos", "Su deuda frente al capital", "Su cuota de mercado"),
             0, "ROE = beneficio neto / patrimonio neto. Un ROE alto y estable suele ser señal de un buen negocio."),
    Question("¿Qué es el ratio de Sharpe?",
             ("Rentabilidad obtenida por cada unidad de riesgo", "El PER ajustado por deuda", "La comisión de un fondo", "El porcentaje de acciones en una cartera"),
             0, "Compara rentabilidad y volatilidad: entre dos inversiones que ganan lo mismo, la de mayor Sharpe lo hizo con menos riesgo."),
    Question("Una 'regla de los dos meses' en el IRPF español afecta a...",
             ("Pérdidas por vender acciones y recomprar las mismas en 2 meses", "El plazo para presentar la renta", "Los dividendos de empresas extranjeras", "Las cuentas remuneradas"),
             0, "Si vendes con pérdidas y recompras valores homogéneos en 2 meses (cotizados), no puedes compensar esa pérdida hasta venderlos."),
    Question("¿Qué es el 'interés compuesto'?",
             ("Ganar intereses también sobre los intereses anteriores", "Un interés fijado por el BCE", "Un préstamo con dos tipos", "Un impuesto sobre el ahorro"),
             0, "Al 7 % anual reinvirtiendo, el dinero se duplica aproximadamente cada 10 años (regla del 72: 72 / 7 ≈ 10)."),
    Question("¿Qué representa el índice IBEX 35?",
             ("Las 35 empresas más líquidas de la bolsa española", "Las 35 empresas más antiguas de España", "Un fondo de inversión público", "El precio de 35 materias primas"),
             0, "Lo revisa un comité técnico cada semestre según liquidez y capitalización."),
    Question("Un fondo indexado con un 0,2 % de comisión frente a uno con un 2 %, durante 30 años...",
             ("La diferencia acumulada puede ser enorme por el interés compuesto", "Da casi igual", "El de 2 % siempre gana más", "Las comisiones no afectan a la rentabilidad"),
             0, "Con un 7 % bruto anual, 10.000 € acaban en ~72.000 € con 0,2 % y en ~43.000 € con 2 % de comisión."),
    Question("¿Qué es la 'deuda neta' de una empresa?",
             ("Deuda financiera menos caja", "Deuda con proveedores", "Deuda que vence este año", "Deuda en moneda extranjera"),
             0, "Se compara a menudo con el EBITDA: deuda neta / EBITDA por encima de 3-4 veces suele considerarse elevada (salvo bancos)."),
    Question("La 'máxima caída' (drawdown) de una inversión mide...",
             ("La mayor bajada desde un máximo hasta un mínimo posterior", "La caída del último día", "Lo que bajó el dividendo", "La diferencia entre compra y venta"),
             0, "Ayuda a saber cuánto 'duele' una inversión en su peor momento: el S&P 500 cayó más de un 50 % en 2008-2009."),
    Question("¿Qué es el 'precio objetivo' de los analistas?",
             ("Su estimación de a dónde puede llegar la cotización en ~12 meses", "El precio mínimo legal de la acción", "El precio de salida a bolsa", "El precio al que recompra la empresa"),
             0, "Es una opinión, no una garantía: los precios objetivo se revisan a menudo y fallan con frecuencia."),
]


def random_question(rng: random.Random | None = None) -> tuple[int, Question]:
    rng = rng or _RNG
    idx = rng.randrange(len(QUESTIONS))
    return idx, QUESTIONS[idx]


def shuffled(question: Question, rng: random.Random | None = None) -> tuple[list[str], int]:
    """Baraja las opciones para que la correcta no sea siempre la primera."""
    rng = rng or _RNG
    order = list(range(len(question.options)))
    rng.shuffle(order)
    return [question.options[i] for i in order], order.index(question.correct)
