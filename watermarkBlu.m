function imgOut = watermarkBlu(img, testo, alphaBlue)
% Inserisce un watermark testuale solo sul canale blu

    % Converte l'immagine in formato double per eseguire operazioni matematiche
    img = im2double(img);

    % Genera un'immagine completamente nera delle stesse dimensioni 
    % dell'originale e vi inserisce il testo del watermark
    wm = insertText(zeros(size(img,1), size(img,2)), [20 20], testo, ...
        'FontSize', 64, 'TextColor', 'white', 'BoxOpacity', 0);

    % Assicura che la matrice del watermark sia a singolo canale (scala di grigi)
    if size(wm,3) == 3
        wm = rgb2gray(im2double(wm));
    else
        wm = im2double(wm);
    end

    % Forza la dimensione del watermark affinché coincida esattamente con l'immagine
    wm = imresize(wm, [size(img,1), size(img,2)]);

    % Inizializza l'immagine di output come copia dell'originale
    imgOut = img;

    % Inietta il watermark nel canale Blu
    imgOut(:,:,3) = min(max(img(:,:,3) + alphaBlue * wm, 0), 1);

    % Riconverte l'immagine finale nel formato standard a 8 bit per il salvataggio
    imgOut = im2uint8(imgOut);
end
