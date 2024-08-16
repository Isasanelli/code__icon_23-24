import rdflib
import os

def create_ontology():
    g = rdflib.Graph()
    baseDir = os.path.dirname(os.path.abspath(__file__))
    
    # Definizione dei namespace per DBpedia e schema.org
    dbpedia_ns = rdflib.Namespace("http://dbpedia.org/ontology/")
    schema_ns = rdflib.Namespace("http://schema.org/")
    
    # Dichiarazione delle classi e proprietà
    g.add((dbpedia_ns.Film, rdflib.RDFS.subClassOf, schema_ns.CreativeWork))
    g.add((dbpedia_ns.TelevisionShow, rdflib.RDFS.subClassOf, schema_ns.CreativeWork))
    g.add((dbpedia_ns.Person, rdflib.RDFS.subClassOf, schema_ns.Person))
    
    # Dichiarazione delle proprietà utilizzando proprietà di DBpedia
    g.add((dbpedia_ns.director, rdflib.RDF.type, rdflib.OWL.ObjectProperty))
    g.add((dbpedia_ns.director, rdflib.RDFS.domain, dbpedia_ns.Film))
    g.add((dbpedia_ns.director, rdflib.RDFS.range, dbpedia_ns.Person))
    
    g.add((dbpedia_ns.starring, rdflib.RDF.type, rdflib.OWL.ObjectProperty))
    g.add((dbpedia_ns.starring, rdflib.RDFS.domain, dbpedia_ns.Film))
    g.add((dbpedia_ns.starring, rdflib.RDFS.range, dbpedia_ns.Person))
    
    g.add((dbpedia_ns.starring, rdflib.RDFS.domain, dbpedia_ns.TelevisionShow))
    g.add((dbpedia_ns.starring, rdflib.RDFS.range, dbpedia_ns.Person))
    
    # Definisce il percorso di output per salvare l'ontologia
    output_dir = os.path.join(baseDir, '..', 'results', 'models', 'knowledge_base')
    output_path = os.path.join(output_dir, 'amazon_prime_ontology.owl')
    
    # Crea la directory se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salvataggio dell'ontologia
    g.serialize(destination=output_path, format='xml')
    
    return g

if __name__ == "__main__":
    ontology = create_ontology()
    print("Ontologia creata e salvata")
