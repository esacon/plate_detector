import re


placa = bool(re.fullmatch(r"(\w{3}[\s\-\*\.]\d{3})", "AAA111   "))

print(placa)

