function imgOut = normalizeImageResolution(imgIn, targetImageSize)
% Uniforma canali e risoluzione delle immagini

    % Verifica che la dimensione target sia stata fornita in input
    if nargin < 2 || isempty(targetImageSize)
        error('Specificare targetImageSize nel formato [altezza larghezza].');
    end

    % Controlla che la dimensione target sia un array di due valori strettamente positivi
    if numel(targetImageSize) ~= 2 || any(targetImageSize <= 0)
        error('targetImageSize deve essere un vettore positivo [altezza larghezza].');
    end

    % Forza la dimensione target ad essere un vettore riga di numeri interi
    targetImageSize = round(double(targetImageSize(:)'));

    % Forza immagine RGB
    if ndims(imgIn) == 2
        imgOut = repmat(imgIn, 1, 1, 3);
    elseif size(imgIn, 3) == 1
        imgOut = repmat(imgIn, 1, 1, 3);
    else
        imgOut = imgIn(:, :, 1:3);
    end

    % Uniforma la risoluzione di tutte le immagini.
    if any([size(imgOut, 1), size(imgOut, 2)] ~= targetImageSize)
        imgOut = imresize(imgOut, targetImageSize);
    end
end
