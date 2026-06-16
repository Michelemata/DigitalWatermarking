function [features, featureNames] = computeWatermarkScore(original, watermarked, bbox)
    
    % Se la bounding box non viene fornita in input, si inizializza come vuota
    if nargin < 3
        bbox = [];
    end

    % Uniforma dimensioni e canali
    if any(size(original,1:2) ~= size(watermarked,1:2))
        watermarked = imresize(watermarked, [size(original,1), size(original,2)]);
    end

    % Si forza la struttura a 3 canali (RGB)
    if size(original,3) == 1
        original = repmat(original, 1, 1, 3);
    end
    if size(watermarked,3) == 1
        watermarked = repmat(watermarked, 1, 1, 3);
    end

    % Conversione nel formato double per eseguire calcoli matematici
    orig = im2double(original);
    wm   = im2double(watermarked);

    % Differenze per canale
    dR = wm(:,:,1) - orig(:,:,1);
    dG = wm(:,:,2) - orig(:,:,2);
    dB = wm(:,:,3) - orig(:,:,3);

    % Estrazione di media e deviazione standard dell'errore per ciascun canale
    redDiffMean   = mean(dR(:));
    redDiffStd    = std(dR(:));
    greenDiffMean = mean(dG(:));
    greenDiffStd  = std(dG(:));
    blueDiffMean  = mean(dB(:));
    blueDiffStd   = std(dB(:));

    % Conversione delle immagini in scala di grigi per calcolare le metriche globali
    grayOrig = rgb2gray(orig);
    grayWm   = rgb2gray(wm);

    % Calcolo della similarità strutturale e dell'errore quadratico medio
    ssimVal = ssim(grayWm, grayOrig);
    mseVal  = immse(grayWm, grayOrig);

    % Calcolo del Peak Signal-to-Noise Ratio (PSNR)
    if mseVal < 1e-12
        psnrVal = 100;
    else
        psnrVal = psnr(grayWm, grayOrig);
    end

    % Calcolo della matrice delle differenze assolute 
    % tra l'immagine originale e quella watermarkata
    absDiff = abs(wm - orig);
    absVec  = absDiff(:);

    % Estrazione di statistiche descrittive sull'errore assoluto globale
    diffMean   = mean(absVec);
    diffStd    = std(absVec);
    diffMedian = median(absVec);
    diffMax    = max(absVec);
    diffP90    = prctile(absVec, 90);
    diffEnergy = mean((wm(:) - orig(:)).^2);

    % Correlazione tra grayscale originale e watermarked
    c = corrcoef(grayOrig(:), grayWm(:));
    if numel(c) >= 2 && isfinite(c(1,2))
        corrGray = c(1,2);
    else
        corrGray = 1;
    end

    % Feature locali sulla ROI (se disponibile)
    if ~isempty(bbox)
        % Verifica e correzione della bounding box per evitare errori di out-of-bounds
        bboxInt = sanitizeBBox(bbox, size(orig));
        if isempty(bboxInt)
            roiOrig = grayOrig;
            roiWm   = grayWm;
        else
            roiOrig = imcrop(grayOrig, bboxInt);
            roiWm   = imcrop(grayWm, bboxInt);
        end
    else
        % Si ritaglia la porzione di immagine interessata
        roiOrig = grayOrig;
        roiWm   = grayWm;
    end

    % Calcolo di SSIM, MSE e PSNR limitatamente alla sola ROI
    ssimROI = ssim(roiWm, roiOrig);
    mseROI = immse(roiWm, roiOrig);
    
    if mseROI < 1e-12
        psnrROI = 100;
    else
        psnrROI = psnr(roiWm, roiOrig);
    end
    
    % Statistiche dell'errore assoluto calcolate solo all'interno della ROI
    roiAbsDiff = abs(roiWm - roiOrig);
    diffMeanROI = mean(roiAbsDiff(:));
    diffStdROI  = std(roiAbsDiff(:));

    % Aggregazione di tutti i valori calcolati in un unico vettore riga
    features = [ ...
        ssimVal, psnrVal, mseVal, ...
        diffMean, diffStd, diffMedian, diffMax, diffP90, diffEnergy, ...
        redDiffMean, redDiffStd, ...
        greenDiffMean, greenDiffStd, ...
        blueDiffMean, blueDiffStd, ...
        corrGray, ...
        ssimROI, psnrROI, diffMeanROI, diffStdROI ...
    ];

    % Se richiesto (nargout > 1), si restituisce un array di celle contenente 
    % i nomi descrittivi associati a ciascuna feature estratta.
    if nargout > 1
        featureNames = { ...
            'ssim', 'psnr', 'mse', ...
            'diffMean', 'diffStd', 'diffMedian', 'diffMax', 'diffP90', 'diffEnergy', ...
            'redDiffMean', 'redDiffStd', ...
            'greenDiffMean', 'greenDiffStd', ...
            'blueDiffMean', 'blueDiffStd', ...
            'corrGray', ...
            'ssimROI', 'psnrROI', 'diffMeanROI', 'diffStdROI' ...
        };
    end
end

% Rende valida e interna all'immagine una bounding box [x y w h]
function bboxOut = sanitizeBBox(bbox, imgSize)

    % Verifica che bounding box contenga esattamente 4 valori numerici validi
    if numel(bbox) ~= 4 || any(~isfinite(bbox))
        bboxOut = [];
        return;
    end

    % Si arrotondano le coordinate e si assicura che partano almeno dal pixel 1
    x = max(1, round(bbox(1)));
    y = max(1, round(bbox(2)));
    w = max(1, round(bbox(3)));
    h = max(1, round(bbox(4)));

    % Si estrapolano le dimensioni massime dell'immagine (Larghezza e Altezza)
    W = imgSize(2);
    H = imgSize(1);

    % Se il punto di origine (x, y) è fuori dall'immagine, si annulla la ROI
    if x > W || y > H
        bboxOut = [];
        return;
    end

    % Si ricalibrano larghezza e altezza in modo 
    % che non sbordino dai margini dell'immagine
    w = min(w, W - x + 1);
    h = min(h, H - y + 1);

    % Se dopo il ridimensionamento larghezza o 
    % altezza sono invalide, si annulla la ROI
    if w < 1 || h < 1
        bboxOut = [];
        return;
    end

    % Si restituisce la bounding box verificata e sicura
    bboxOut = [x, y, w, h];
end
