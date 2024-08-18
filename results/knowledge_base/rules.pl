recommend_for_kids(Title) :- rating(Title, 'G'), write('Raccomandato per bambini: '), write(Title), nl.
recommend_for_teens(Title) :- rating(Title, 'PG-13'), write('Raccomandato per adolescenti: '), write(Title), nl.
recommend_for_adults(Title) :- rating(Title, 'R'), write('Raccomandato per adulti: '), write(Title), nl.
recommend_for_family(Title) :- rating(Title, 'PG'), preference_for('Family Features', Title), write('Raccomandato per famiglie: '), write(Title), nl.
recommend_for_action_lovers(Title) :- preference_for('Action & Adventure', Title), write('Raccomandato per gli amanti dell\'azione: '), write(Title), nl.
