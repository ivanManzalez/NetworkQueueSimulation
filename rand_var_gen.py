# 𝒙 = − (𝟏/lambda) 𝒍𝒏(𝟏 − 𝑼)
from numpy import log
import random


def rand_var_generator(lamda):
  U = random.random()
  return -(1/lamda)*log(1-U)