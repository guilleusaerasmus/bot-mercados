#!/usr/bin/env bash
# Instala (o actualiza) el Bot de Mercados en un servidor Ubuntu como servicio que arranca solo
# y se reinicia si falla. Pensado para Oracle Cloud "Always Free" (ARM), vale para cualquier Ubuntu 22.04+.
#
#   bash instalar.sh            # primera instalación o actualización
#
# El archivo .env (con las claves) NO está en GitHub: hay que copiarlo a ~/bot-mercados/.env.
set -euo pipefail

REPO="${REPO:-https://github.com/guilleusaerasmus/bot-mercados.git}"
DIR="$HOME/bot-mercados"
VENV="$HOME/.venvs/bot-mercados"
SERVICE=bot-mercados

echo "==> Paquetes del sistema"
sudo apt-get update -y -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git

echo "==> Código"
if [ -d "$DIR/.git" ]; then
    git -C "$DIR" pull --ff-only
else
    git clone "$REPO" "$DIR"
fi

echo "==> Entorno de Python y dependencias (la primera vez tarda unos minutos)"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$DIR/requirements.txt"

echo "==> Servicio systemd"
sudo tee "/etc/systemd/system/$SERVICE.service" >/dev/null <<EOF
[Unit]
Description=Bot de Mercados (Telegram)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$DIR
ExecStart=$VENV/bin/python -m bot_mercados telegram
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable -q "$SERVICE"

echo "==> Copia de seguridad diaria de la base de datos (usuarios, pagos, carteras), 14 días"
mkdir -p "$HOME/copias"
BACKUP="$VENV/bin/python -c \"import sqlite3,datetime,pathlib;h=pathlib.Path.home();s=h/'.bot-mercados'/'bot.db';d=h/'copias'/f'bot-{datetime.date.today()}.db';s.exists() and sqlite3.connect(s).backup(sqlite3.connect(d))\" && find $HOME/copias -name 'bot-*.db' -mtime +14 -delete"
( crontab -l 2>/dev/null | grep -v 'bot-mercados-copia' ; echo "30 3 * * * $BACKUP # bot-mercados-copia" ) | crontab -

if [ -f "$DIR/.env" ]; then
    sudo systemctl restart "$SERVICE"
    sleep 5
    systemctl --no-pager --lines=5 status "$SERVICE" || true
    echo "==> Listo. Registro en directo: journalctl -u $SERVICE -f"
else
    echo "==> Instalado, pero falta $DIR/.env con las claves. Cópialo y ejecuta: sudo systemctl restart $SERVICE"
fi
