from SPARQLWrapper import SPARQLWrapper, JSON

def query_dbpedia(title):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(f"""
    SELECT ?abstract WHERE {{
        ?film rdfs:label "{title}"@en .
        ?film dbo:abstract ?abstract .
        FILTER (lang(?abstract) = 'en')
    }}
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        print(result["abstract"]["value"])

# Esempio di query
query_dbpedia("The Matrix")
