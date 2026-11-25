"""Pares planejados para a ablação."""
PARES=[('abordagem','espacial_aumentado','hibrido_aumentado'),('abordagem','espacial_aumentado','hibrido_aumentado_ampliado'),('ampliacao_espectral','hibrido_aumentado','hibrido_aumentado_ampliado')]
for representacao in ('espacial','espectral','hibrido'):
    PARES.append(('aumento',representacao+'_original',representacao+'_aumentado'))
for representacao in ('espectral','hibrido'):
    PARES.append(('aumento',representacao+'_original_ampliado',representacao+'_aumentado_ampliado'))
    for regime in ('original','aumentado'):
        PARES.append(('ampliacao_espectral',representacao+'_'+regime,representacao+'_'+regime+'_ampliado'))
for regime in ('original','aumentado'):
    for rep in ('','_ampliado'):
        par=('combinacao','espacial_'+regime,'hibrido_'+regime+rep)
        if par[1:] not in [(x[1],x[2]) for x in PARES]: PARES.append(par)
