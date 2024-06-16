import torch
from utils.logging import logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Identified device: {}", device)
