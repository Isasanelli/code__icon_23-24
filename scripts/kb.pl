% Fatti generati automaticamente dai dati
% Esempio:
% rating('The Witcher', 'TV-MA').
% type('The Witcher', 'TV Show - Action & Adventure').
% director_of('John Doe', 'The Witcher').
% classify('The Witcher', 'Action & Adventure').
% preference_for('Action & Adventure', 'The Witcher').

% Regole generali di raccomandazione basate sul rating (età) e preferenze del pubblico

recommend_for_kids(Title) :-
    rating(Title, 'G'),
    write('Raccomandato per bambini: '), write(Title), nl.

recommend_for_teens(Title) :-
    rating(Title, 'PG-13'),
    write('Raccomandato per adolescenti: '), write(Title), nl.

recommend_for_adults(Title) :-
    rating(Title, 'R'),
    write('Raccomandato per adulti: '), write(Title), nl.

recommend_for_family(Title) :-
    rating(Title, 'PG'),
    preference_for('Family Features', Title),
    write('Raccomandato per famiglie: '), write(Title), nl.

recommend_for_action_lovers(Title) :-
    preference_for('Action & Adventure', Title),
    write('Raccomandato per gli amanti dell\'azione: '), write(Title), nl.

% Regole per la raccomandazione basata su attributi specifici

% Raccomanda un film basato sul regista
recommend_based_on_director(Director) :-
    director_of(Director, Title),
    format('Raccomandato per chi ama i film di ~w: ~w~n', [Director, Title]).

% Raccomanda un film basato sul genere
recommend_based_on_genre(Genre) :-
    classify(Title, Genre),
    format('Raccomandato per chi ama i film di genere ~w: ~w~n', [Genre, Title]).

% Raccomanda un film basato sulla preferenza specifica
recommend_based_on_preference(Preference) :-
    preference_for(Preference, Title),
    format('Raccomandato per chi ha una preferenza per ~w: ~w~n', [Preference, Title]).

% Raccomandazioni multiple basate su attributi

% Raccomanda tutti i film di un certo regista
recommend_all_by_director(Director) :-
    findall(Title, director_of(Director, Title), Titles),
    write('Tutti i film di '), write(Director), write(': '), nl,
    print_titles(Titles).

% Raccomanda tutti i film di un certo genere
recommend_all_by_genre(Genre) :-
    findall(Title, classify(Title, Genre), Titles),
    write('Tutti i film di genere '), write(Genre), write(': '), nl,
    print_titles(Titles).

% Raccomanda tutti i film basati su una preferenza specifica
recommend_all_by_preference(Preference) :-
    findall(Title, preference_for(Preference, Title), Titles),
    write('Tutti i film per la preferenza '), write(Preference), write(': '), nl,
    print_titles(Titles).

% Regola di supporto per stampare una lista di titoli
print_titles([]) :- !.
print_titles([Head|Tail]) :-
    write('- '), write(Head), nl,
    print_titles(Tail).
