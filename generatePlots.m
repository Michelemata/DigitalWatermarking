function generatePlots( ...
    XA_test, YA_test, ...
    XB_test, YB_test, ...
    featureNames, modelRF_A, modelRF_B, ...
    imds, outputA, outputB, bboxList, globalTestIdx, figuresDir, targetImageSize)

    % Se la dimensione target per le immagini non 
    % viene fornita, la si inizializza come vuota
    if nargin < 14 || isempty(targetImageSize)
        targetImageSize = [];
    end

    %% GESTIONE DIFFERENZA COLONNE TRA A e B
    featureNamesA = featureNames;
    featureNamesB = featureNames;

    % Se XB_test ha meno colonne, eliminiamo le feature ROI note dalla sua lista nomi
    if size(XB_test, 2) < size(XA_test, 2)
        idxToRemove = ismember(featureNamesB, {'ssimROI', 'psnrROI', 'diffMeanROI', 'diffStdROI'});
        featureNamesB(idxToRemove) = [];
    end

    %% 1) Boxplot di feature chiave sul test set

    % Si definisce un set di feature particolarmente rilevanti da visualizzare
    desiredNames = {'blueDiffMean', 'diffEnergy', 'ssim', 'ssimROI'};
    % Si cercano gli indici per SVD (A)
    idxSelA = [];
    selNamesA = {};
    for k = 1:numel(desiredNames)
        idx = find(strcmp(featureNamesA, desiredNames{k}), 1);
        if ~isempty(idx)
            idxSelA(end+1) = idx; %#ok<AGROW>
            selNamesA{end+1} = featureNamesA{idx}; %#ok<AGROW>
        end
    end

    % Si cercano gli indici per Blu (B) - filtrando le feature inesistenti
    idxSelB = [];
    selNamesB = {};
    for k = 1:numel(desiredNames)
        idx = find(strcmp(featureNamesB, desiredNames{k}), 1);
        if ~isempty(idx)
            idxSelB(end+1) = idx; %#ok<AGROW>
            selNamesB{end+1} = featureNamesB{idx}; %#ok<AGROW>
        end
    end

    % Generazione dei boxplot
    if ~isempty(idxSelA)
        createBoxplotFigure(XA_test(:, idxSelA), YA_test, selNamesA, ...
            'Feature distributions - Watermark SVD', ...
            fullfile(figuresDir, 'boxplot_feature_svd.png'));
    end

    if ~isempty(idxSelB)
        createBoxplotFigure(XB_test(:, idxSelB), YB_test, selNamesB, ...
            'Feature distributions - Watermark Blu', ...
            fullfile(figuresDir, 'boxplot_feature_blu.png'));
    end


    %% 2) Feature importance dalla Random Forest

    % Si estrae l'importanza di ciascuna variabile calcolata dal modello RF
    impA = predictorImportance(modelRF_A);
    impB = predictorImportance(modelRF_B);

    % Generazione dei grafici a barre orizzontali
    createFeatureImportanceFigure(impA, featureNamesA, ...
        'Feature importance - Random Forest - SVD', ...
        fullfile(figuresDir, 'feature_importance_svd.png'));

    createFeatureImportanceFigure(impB, featureNamesB, ...
        'Feature importance - Random Forest - Blu', ...
        fullfile(figuresDir, 'feature_importance_blu.png'));

    %% 3) Esempi visivi: originale, watermarked, heatmap differenze

    % Si cerca la prima immagine del test set che appartiene alla classe "Watermarked"
    posTestLocal = find(YA_test == 1, 1, 'first');
    if ~isempty(posTestLocal)
        globalIdx = globalTestIdx(posTestLocal);

        % Genera un pannello visivo che confronta l'immagine originale,
        % quella modificata e le relative heatmap degli errori
        createVisualExample(imds, outputA, globalIdx, bboxList{globalIdx}, targetImageSize, ...
            'Visual example - SVD watermark', ...
            fullfile(figuresDir, 'visual_example_svd.png'));
        createVisualExample(imds, outputB, globalIdx, bboxList{globalIdx}, targetImageSize, ...
            'Visual example - Blue-channel watermark', ...
            fullfile(figuresDir, 'visual_example_blu.png'));
    end

end


%% FUNZIONI HELPER

% Crea e salva una griglia di boxplot
function createBoxplotFigure(X, Y, featureNames, figTitle, savePath)
    nFeat = numel(featureNames);
    nCols = 2;
    nRows = ceil(nFeat / nCols);

    f = figure('Name', figTitle, 'Position', [100 100 1200 600]);
    t = tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(t, figTitle);

    % Si categorizzano le etichette per una corretta formattazione dell'asse X
    groupLabels = categorical(Y, [0 1], {'Original', 'Watermarked'});

    % Si itera su ogni feature per tracciare il singolo boxplot nel relativo riquadro
    for i = 1:nFeat
        nexttile;
        boxchart(groupLabels, X(:, i));
        xlabel('Class');
        ylabel(featureNames{i}, 'Interpreter', 'none');
        title(featureNames{i}, 'Interpreter', 'none');
        grid on;
    end

    saveas(f, savePath);
end


% Crea un grafico a barre orizzontali delle top feature
function createFeatureImportanceFigure(importance, featureNames, figTitle, savePath)

    % Si ordinano le feature per importanza in modo decrescente
    [sortedImp, idx] = sort(importance(:), 'descend');

    % Si selezionano al massimo le prime 10 feature più rilevanti
    topK = min(10, numel(sortedImp));
    idx = idx(1:topK);
    sortedImp = sortedImp(1:topK);
    sortedNames = featureNames(idx);

    f = figure('Name', figTitle, 'Position', [100 100 900 650]);

    % Si inverte l'ordine per avere la feature più importante in alto nel grafico
    barh(sortedImp(end:-1:1));
    yticks(1:topK);
    yticklabels(sortedNames(end:-1:1));
    xlabel('Importance');
    title(figTitle);
    grid on;

    saveas(f, savePath);
end


% Crea un confronto visivo con mappe di calore delle differenze
function createVisualExample(imds, outputFolder, globalIdx, bbox, targetImageSize, figTitle, savePath)
    
    % Caricamento delle immagini originali e watermarkate
    original = imread(imds.Files{globalIdx});

    % Normalizzazione delle dimensioni delle immagini
    if ~isempty(targetImageSize)
        original = normalizeImageResolution(original, targetImageSize);
    end
    [~, name, ext] = fileparts(imds.Files{globalIdx});
    watermarked = imread(fullfile(outputFolder, [name, ext]));
    if ~isempty(targetImageSize)
        watermarked = normalizeImageResolution(watermarked, targetImageSize);
    end

    % Si convertono in scala di grigi per calcolare la differenza assoluta
    if size(original,3) == 1
        originalGray = original;
    else
        originalGray = rgb2gray(original);
    end
    if size(watermarked,3) == 1
        watermarkedGray = watermarked;
    else
        watermarkedGray = rgb2gray(watermarked);
    end

    % Calcolo della matrice delle differenze assolute
    absDiff = imabsdiff(originalGray, watermarkedGray);

    % Disegna la bounding box sull'immagine per evidenziare la ROI
    if ~isempty(bbox)
        originalBox = insertShape(original, 'Rectangle', bbox, 'LineWidth', 3);
        watermarkedBox = insertShape(watermarked, 'Rectangle', bbox, 'LineWidth', 3);
    else
        originalBox = original;
        watermarkedBox = watermarked;
    end

    % Creazione di un layout 2x2 per i subplot
    f = figure('Name', figTitle, 'Position', [50 50 1200 800]);
    t = tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(t, figTitle);

    % Riquadro 1: Immagine originale
    nexttile;
    imshow(originalBox);
    title('Original image');

    % Riquadro 2: Immagine watermarkata
    nexttile;
    imshow(watermarkedBox);
    title('Watermarked image');

    % Riquadro 3: Heatmap della differenza sull'intera immagine
    nexttile;
    imshow(absDiff, []);
    title('Absolute difference heatmap');
    colorbar;

    % Riquadro 4: Heatmap concentrata sulla sola ROI (se esistente)
    nexttile;
    if ~isempty(bbox)
        roiOrig = imcrop(originalGray, sanitizeBBox(bbox, size(original)));
        roiWm = imcrop(watermarkedGray, sanitizeBBox(bbox, size(watermarked)));
        roiDiff = imabsdiff(roiOrig, roiWm);
        imshow(roiDiff, []);
        title('ROI difference');
        colorbar;
    else
        imshow(absDiff, []);
        title('Full-image difference');
        colorbar;
    end

    saveas(f, savePath);
end

% Rende valida e interna all'immagine una bounding box [x y w h]
function bboxOut = sanitizeBBox(bbox, imgSize)

    % Verifica preliminare sui dati in ingresso
    if numel(bbox) ~= 4 || any(~isfinite(bbox))
        bboxOut = [];
        return;
    end

    % Arrotondamento delle coordinate per l'estrazione dei pixel
    x = max(1, round(bbox(1)));
    y = max(1, round(bbox(2)));
    w = max(1, round(bbox(3)));
    h = max(1, round(bbox(4)));

    W = imgSize(2);
    H = imgSize(1);

    % Se il punto di ancoraggio è fuori dall'immagine, si invalida la box
    if x > W || y > H
        bboxOut = [];
        return;
    end

    % Ridimensionamento di larghezza e altezza per non uscire dai bordi
    w = min(w, W - x + 1);
    h = min(h, H - y + 1);

    if w < 1 || h < 1
        bboxOut = [];
        return;
    end

    bboxOut = [x, y, w, h];
end
