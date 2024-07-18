import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Creazione della rete bayesiana
model = BayesianNetwork([('Genere', 'Recensione'), ('Attori', 'Recensione')])

# Definizione delle distribuzioni di probabilità
from pgmpy.factors.discrete import TabularCPD
cpd_genere = TabularCPD(variable='Genere', variable_card=3, values=[[0.2], [0.5], [0.3]])
cpd_attori = TabularCPD(variable='Attori', variable_card=2, values=[[0.6], [0.4]])
cpd_recensione = TabularCPD(variable='Recensione', variable_card=2, values=[[0.1, 0.4, 0.3, 0.8, 0.7, 0.9],
                                                                            [0.9, 0.6, 0.7, 0.2, 0.3, 0.1]],
                            evidence=['Genere', 'Attori'], evidence_card=[3, 2])

model.add_cpds(cpd_genere, cpd_attori, cpd_recensione)

# Inferenza
infer = VariableElimination(model)
posterior_p = infer.map_query(variables=['Recensione'], evidence={'Genere': 2, 'Attori': 1})
print(posterior_p)
