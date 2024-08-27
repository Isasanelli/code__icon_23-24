% Regole per la raccomandazione dei contenuti

recommend(Content) :-
    content(Content, _, _, Preference),
    Preference >= 80.

similar_content(Title1, Title2) :-
    content(Title1, Category, _, _),
    content(Title2, Category, _, _),
    Title1 \= Title2.

recommend_similar(Title, RecommendedTitle) :-
    similar_content(Title, RecommendedTitle),
    content(RecommendedTitle, _, _, Preference),
    Preference >= 60.

recommend_by_genre(Genre, Content) :-
    content(Content, Genre, _, Preference),
    Preference >= 50.

recommend_by_year(Year, Content) :-
    content(Content, _, Year, Preference),
    Preference >= 50.

recommend_by_genre_year(Genre, Year, Content) :-
    content(Content, Genre, Year, Preference),
    Preference >= 50.

