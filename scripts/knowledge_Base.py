import rdflib
import os

def create_ontology():
    g = rdflib.Graph()
    baseDir = os.path.dirname(os.path.abspath(__file__))
    
    # Definizione di un namespace realistico per Amazon Prime
    amazon_ns = rdflib.Namespace("http://amazon.com/prime#")
    schema_ns = rdflib.Namespace("http://schema.org/")
    
    # Dichiarazione delle classi e proprietà
    g.add((amazon_ns.Movie, rdflib.RDFS.subClassOf, schema_ns.CreativeWork))
    g.add((amazon_ns.TVShow, rdflib.RDFS.subClassOf, schema_ns.CreativeWork))
    g.add((amazon_ns.Actor, rdflib.RDFS.subClassOf, schema_ns.Person))
    g.add((amazon_ns.Director, rdflib.RDFS.subClassOf, schema_ns.Person))
    
    # Dichiarazione delle proprietà utilizzando proprietà simili a quelle di Schema.org
    g.add((amazon_ns.directedBy, rdflib.RDF.type, rdflib.OWL.ObjectProperty))
    g.add((amazon_ns.directedBy, rdflib.RDFS.domain, amazon_ns.Movie))
    g.add((amazon_ns.directedBy, rdflib.RDFS.range, amazon_ns.Director))
    
    g.add((amazon_ns.starring, rdflib.RDF.type, rdflib.OWL.ObjectProperty))
    g.add((amazon_ns.starring, rdflib.RDFS.domain, amazon_ns.Movie))
    g.add((amazon_ns.starring, rdflib.RDFS.range, amazon_ns.Actor))
    
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
