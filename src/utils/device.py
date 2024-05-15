import torch
from utils.logging import logger


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logger.info("Identified device: {}", device)
