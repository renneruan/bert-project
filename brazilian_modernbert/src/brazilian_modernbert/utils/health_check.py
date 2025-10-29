import subprocess
import logging

import torch

logger = logging.getLogger(__name__)


def health_check():

    commands = [
        ["hostname"],
        ["rocm-smi"],
        ["pip", "list"],
    ]

    for command in commands:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=True
            )

            logger.info(result.stdout)
            if result.stderr:
                logger.info(result.stderr)

        except FileNotFoundError:
            logger.error(f"Error: Command '{command[0]}' not found.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error executing command: {e}")
            logger.error(e.stdout)
            logger.error(e.stderr)

    # Verificar tamanho e GPUs disponíveis
    logger.info(torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        logger.info(torch.cuda.get_device_properties(i))
