from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

def train_ensemble_model(X, y):
    rf = RandomForestClassifier(random_state=42)
    ada = AdaBoostClassifier(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    knn = KNeighborsClassifier()
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('ada', ada), ('dt', dt), ('knn', knn)],
        voting='soft'
    )
    
    ensemble.fit(X, y)
    return ensemble