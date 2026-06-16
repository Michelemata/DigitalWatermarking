function scoresPositive = extractPositiveClassScores(scores)
% Estrae lo score della classe positiva.

    % Se l'array di input è vuoto, si restituisce un vettore 
    % vuoto e si interrompe l'esecuzione della funzione.
    if isempty(scores)
        scoresPositive = [];
        return;
    end

    % Se l'input è una matrice, si isola e si restituisce 
    % la seconda colonna (Classe 1 = Watermarked).
    if size(scores, 2) > 1
        scoresPositive = scores(:, 2);
    else
        % Se l'input è un vettore singolo, lo si mantiene inalterato
        scoresPositive = scores(:);
    end
end
