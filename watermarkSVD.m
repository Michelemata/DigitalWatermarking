function imgOut = watermarkSVD(img, detector, testo, bbox, alphaSVD)
% Applica un watermark SVD sulla luminanza (canale Y)
   
    % Forza RGB se l'immagine è in scala di grigi
    if size(img,3) == 1
        img = repmat(img, 1, 1, 3);
    end

    % Rilevazione ROI con YOLO solo se la bbox non e' stata gia' calcolata
    if nargin < 4 || isempty(bbox)
        [bboxes, scores] = detect(detector, img);

        % Se YOLO non trova nulla, ritorna immagine invariata
        if isempty(bboxes)
            imgOut = img;
            return;
        end

        % Sceglie la bounding box con score massimo
        if isempty(scores)
            bestIdx = 1;
        else
            [~, bestIdx] = max(scores);
        end
        bbox = bboxes(bestIdx, :);
    end

    % Se la bbox è vuota, esce dalla funzione
    if isempty(bbox)
        imgOut = img;
        return;
    end

    % Arrotonda e limita la bbox ai bordi dell'immagine
    x1 = max(1, round(bbox(1)));
    y1 = max(1, round(bbox(2)));
    w  = max(1, round(bbox(3)));
    h  = max(1, round(bbox(4)));

    [H, W, ~] = size(img);
    x2 = min(W, x1 + w - 1);
    y2 = min(H, y1 + h - 1);

    % Se le coordinate risultano invalide, esce dalla funzione
    if x2 <= x1 || y2 <= y1
        imgOut = img;
        return;
    end

    % Estrai ROI direttamente con indicizzazione per evitare problemi di imcrop
    roi = img(y1:y2, x1:x2, :);

    % Conversione in YCbCr: si lavora solo sulla luminanza Y
    roiYCbCr = rgb2ycbcr(roi);
    Y  = im2double(roiYCbCr(:,:,1));
    Cb = roiYCbCr(:,:,2);
    Cr = roiYCbCr(:,:,3);

    % SVD sulla luminanza
    [U, S, V] = svd(Y);

    % Creazione watermark testuale come maschera in [0,1]
    wmCanvas = zeros(size(Y), 'uint8');
    wmRGB = insertText(repmat(wmCanvas, [1 1 3]), [20 20], testo, ...
        'FontSize', 64, ...
        'TextColor', 'white', ...
        'BoxOpacity', 0);

    wm = rgb2gray(im2double(wmRGB));
    wm = imresize(wm, size(Y));

    % Inserimento del watermark nel blocco diagonale di S
    d = min(size(S));
    S(1:d, 1:d) = S(1:d, 1:d) + alphaSVD * wm(1:d, 1:d);

    % Ricostruzione della sola luminanza
    Yw = U * S * V';
    Yw = min(max(Yw, 0), 1);

    % Rimonta ROI mantenendo Cb e Cr invariati
    roiYCbCrW = cat(3, im2uint8(Yw), Cb, Cr);
    roiWatermarkedRGB = ycbcr2rgb(roiYCbCrW);

    % Inserisci la ROI modificata nell'immagine originale
    imgOut = img;
    imgOut(y1:y2, x1:x2, :) = roiWatermarkedRGB;
end
