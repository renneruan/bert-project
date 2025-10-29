"""
Módulo de configurações do mecanismo de logging para o projeto.

Apresenta uma função única, 'setup_logging' que deve ser chamada no entry point
da aplicação, seja o main.py ou api.py.

Salva os logs no arquivo logs/app.log.
"""

import os
import sys
import logging

# Formato padrão e legível para as mensagens de log.
# Ex: [2025-09-14 17:10:00] - INFO - main - Mensagem do log.
LOG_FORMAT = "[%(asctime)s] - %(levelname)s - %(module)s - %(message)s"


def setup_logging(log_dir="logs"):
    """
    Configura o sistema de logging de forma centralizada.

    Esta função deve ser chamada uma única vez no ponto de entrada da aplicação.
    Ela configura o logging para enviar mensagens tanto para um arquivo
    quanto para o console (terminal).

    :param log_dir: Diretório onde o arquivo de log será salvo.
    """
    log_filepath = os.path.join(log_dir, "app.log")
    os.makedirs(log_dir, exist_ok=True)

    # Configura o logger raiz do projeto.
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(
                log_filepath, encoding="utf-8"
            ),  # Handler para salvar em arquivo.
            logging.StreamHandler(
                sys.stdout
            ),  # Handler para exibir no terminal.
        ],
    )

    logging.info("Sistema de logging configurado com sucesso.")
