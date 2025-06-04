from eexp_engine import client
import eexp_config

# EXPERIMENTO BINARIO
exp_name = 'main_binary'
client.run(__file__, exp_name, eexp_config)

# EXPERIMENTO MULTICLASE
# exp_name = 'main_multiclass'
# client.run(__file__, exp_name, eexp_config)
