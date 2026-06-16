function imgNoisy = addNoise(img, enableNoise, noiseType, noiseVariance, noiseDensity)
    
    % Inizializzazione dell'immagine di output
    imgNoisy = img;

    % Viene restituita l'immagine originale se enableNoise è falsa
    if nargin < 2 || ~enableNoise
        return;
    end

    % Viene restituita l'immagine originale se il rumore
    % è impostato su 'none' o non è specificato
    if nargin < 3 || isempty(noiseType) || strcmpi(noiseType, 'none')
        return;
    end

    % Si seleziona e si applica la degradazione in base al tipo di rumore richiesto
    switch lower(noiseType)
        case 'gaussian'
            % Se la varianza non è specificata, si assegna un valore di default
            if nargin < 4 || isempty(noiseVariance)
                noiseVariance = 0.01;
            end
            % Applicazione del rumore Gaussiano (media 0, varianza specificata)
            imgNoisy = imnoise(imgNoisy, 'gaussian', 0, noiseVariance);

        case 'salt & pepper'
            % Se la densità non è specificata, si assegna un valore di default
            if nargin < 5 || isempty(noiseDensity)
                noiseDensity = 0.01;
            end
            % Applicazione del rumore impulsivo
            imgNoisy = imnoise(imgNoisy, 'salt & pepper', noiseDensity);

        otherwise
            error('Tipo di rumore non riconosciuto: %s', noiseType);
    end
end
