function bbox = selectBestBoundingBox(bboxes, scores)
% Seleziona la bounding box piu' rilevante

    % Inizializzazione della variabile di output come array vuoto
    bbox = [];

    % Interrompe la funzione se non viene rilevato alcun oggetto
    if isempty(bboxes)
        return;
    end

    % Seleziona l'oggetto con l'area maggiore se 
    % gli score non vengono passati o sono vuoti
    if nargin < 2 || isempty(scores)
        [~, idx] = max(bboxes(:,3) .* bboxes(:,4));
        bbox = bboxes(idx,:);
        return;
    end
    
    % Assicura che gli score siano formattati come vettore colonna
    scores = scores(:);

    % Sceglie l'area più grande se il numero di score non 
    % coincide con il numero di bounding box trovate
    if numel(scores) ~= size(bboxes,1)
        [~, idx] = max(bboxes(:,3) .* bboxes(:,4));
        bbox = bboxes(idx,:);
        return;
    end
    
    % Calcola le aree di tutte le bounding box rilevate
    areas = bboxes(:,3) .* bboxes(:,4);

    % Seleziona la bounding box migliore massimizzando lo score
    [~, idx] = max(scores + 1e-6 * normalizeVector(areas));
    bbox = bboxes(idx,:);
end

%% FUNZIONI HELPER

% Esegue una normalizzazione Min-Max
function y = normalizeVector(x)

    % Converte l'input in un vettore colonna di tipo double
    x = double(x(:));

    if all(x == x(1))
        y = zeros(size(x));
    else
        % Applicazione della formula di normalizzazione Min-Max
        y = (x - min(x)) ./ (max(x) - min(x));
    end
end
