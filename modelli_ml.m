%% PIPELINE PRINCIPALE: Confronto Watermark SVD vs Watermark Blu

% Pulizia dell'ambiente di lavoro
% In modalità normale il file esegue una singola configurazione.
% In modalità sweep, invece, le variabili sweepMode, trialNumber, sweepIdxValue 
% e sweepAlphaValue vengono passate dallo script runSweepAlphaIdx.m
if ~exist('sweepMode','var') || ~sweepMode
    clc; clear; close all;
    rng(42);

    % Configurazione singola di default, usata quando si lancia direttamente
    % modelli_ml.m senza eseguire lo sweep automatico.
    sweepMode = false;
    trialNumber = 1;
    sweepIdxValue = 0.25;
    sweepAlphaValue = 0.02;
else
    close all;
    % Fissa il seed per la riproducibilità, variandolo leggermente tra i trial
    rng(42 + trialNumber);
end


%% Caricamento dataset

DatasetPath = 'Cars_1999_2001';
imds = imageDatastore(DatasetPath);

% Conta quante immagini sono presenti all'interno del dataset
numImages = numel(imds.Files);

% Normalizzazione della risoluzione per uniformità
targetImageSize = [240 360];

%% Parametri sperimentali idx / alpha
% Definisce la percentuale di immagini a cui applicare i watermark
watermarkRatio = sweepIdxValue;

% alphaSVD e alphaBlue controllano l'intensità del watermark nelle due tecniche
alphaSVD = sweepAlphaValue;
alphaBlue = sweepAlphaValue;

% Tag del trial, usato per separare cartelle e risultati delle diverse
% configurazioni senza sovrascrivere i grafici già prodotti
trialTag = sprintf('idx_%0.2f_alpha_%0.2f', watermarkRatio, sweepAlphaValue);
trialTag = strrep(trialTag, '.', '_');

fprintf('Configurazione trial: idx = %.2f | alphaSVD = %.3f | alphaBlue = %.3f\n', ...
    watermarkRatio, alphaSVD, alphaBlue);

%% Gestione Rumore
% Se enableNoise = true, il rumore viene applicato a tutte le immagini,
% sia quelle con il watermark che quelle originali
enableNoise = true;
noiseType = 'gaussian';      % 'none', 'gaussian', 'salt & pepper'
noiseVariance = 0.01;        % usato se noiseType = 'gaussian'
noiseDensity = 0.01;         % usato se noiseType = 'salt & pepper'

% Generazione dinamica del tag identificativo
if ~enableNoise || strcmpi(noiseType, 'none')
    noiseTag = 'clean';
elseif strcmpi(noiseType, 'gaussian')
    noiseTag = sprintf('gaussian_%g', noiseVariance);
elseif strcmpi(noiseType, 'salt & pepper')
    noiseTag = sprintf('saltpepper_%g', noiseDensity);
else
    error('Tipo di rumore non riconosciuto: %s', noiseType);
end
noiseTag = matlab.lang.makeValidName(strrep(noiseTag, '.', '_'));

%% Cartelle di output

% Se si esegue lo sweep, ogni combinazione idx/alpha viene salvata in una
% sottocartella dedicata, così i risultati non vengono sovrascritti
if sweepMode
    outputTag = [noiseTag, '_', trialTag];
    figuresDir = fullfile('Results', noiseTag, 'sweep_alpha_idx', ...
        trialTag);
else
    outputTag = noiseTag;
    figuresDir = fullfile('Results', noiseTag);
end

% Definisce i nomi delle cartelle dove salverà i risultati e i grafici
outputA = ['dataset_A_', outputTag];
outputB = ['dataset_B_', outputTag];

% Crea fisicamente le cartelle sul disco se non esistono già
if ~exist(outputA, 'dir'); mkdir(outputA); end
if ~exist(outputB, 'dir'); mkdir(outputB); end
if ~exist(figuresDir, 'dir'); mkdir(figuresDir); end

% Stampa un messaggio sulla console per confermare quale scenario sta eseguendo
fprintf('Scenario rumore: %s | output: %s, %s\n', noiseTag, outputA, outputB);
fprintf('Cartella figure trial: %s\n', figuresDir);

%% Detector YOLO pre-addestrato

% Inizializza YOLOv8
detector = yolov8ObjectDetector('yolov8s');

% Misura e registra il tempo necessario per caricare il modello in memoria
detectorCreationTimer = tic;
detectorInitTime = toc(detectorCreationTimer);
detectorType = class(detector);

%% Selezione immagini da watermarkare

% Definisce la percentuale di immagini a cui applicare i watermark
idxSubset = randperm(numImages, round(watermarkRatio * numImages));

% Memorizza la bbox scelta per ogni immagine per riutilizzarla nelle feature ROI
bboxList = cell(numImages, 1);

%% Generazione dataset A e B

% Si preallocano i vettori per registrare i tempi di esecuzione delle 
% singole fasi, ottimizzando la misurazione delle performance
timePreprocessingGeneration = zeros(numImages, 1);
timeYOLODetection = zeros(numImages, 1);
timeWatermarkSVD = zeros(numImages, 1);
timeWatermarkBlu = zeros(numImages, 1);
timeImageSaving = zeros(numImages, 1);
numDetections = zeros(numImages, 1);

% Avvio timer
totalGenerationTimer = tic;

for i = 1:numImages
    % Lettura e normalizzazione della risoluzione dell'immagine corrente
    tStep = tic;
    img = imread(imds.Files{i});
    img = normalizeImageResolution(img, targetImageSize);
    timePreprocessingGeneration(i) = toc(tStep);

    [~, name, ext] = fileparts(imds.Files{i});
    fileName = [name, ext];
    
    % Esecuzione della detection per individuare gli oggetti nell'immagine
    tStep = tic;
    [bboxes, scores] = detect(detector, img);
    timeYOLODetection(i) = toc(tStep);
    numDetections(i) = size(bboxes, 1);

    % Seleziona e memorizza la bounding box con il punteggio più alto
    bestBBox = selectBestBoundingBox(bboxes, scores);
    bboxList{i} = bestBBox;

    % Verifica se l'immagine corrente appartiene al sottoinsieme da
    % watermarkare
    if ismember(i, idxSubset)
        tStep = tic;
        % Applicazione del watermark SVD sfruttando la bounding box
        imgSVD = watermarkSVD(img, detector, 'Michele', bestBBox, alphaSVD);

        % Aggiunta del rumore dopo il watermark (se enableNoise = true)
        imgSVD = addNoise(imgSVD, enableNoise, noiseType, noiseVariance, noiseDensity);
        timeWatermarkSVD(i) = toc(tStep);

        tStep = tic;

        % Applicazione del watermark Blu 
        imgBlu = watermarkBlu(img, 'Valeria', alphaBlue);

        % Aggiunta del rumore dopo il watermark (se enableNoise = true)
        imgBlu = addNoise(imgBlu, enableNoise, noiseType, noiseVariance, noiseDensity);
        timeWatermarkBlu(i) = toc(tStep);
    else
        % Alle immagini non modificate viene comunque applicato il rumore 
        % (se abilitato) per non sbilanciare l'apprendimento del classificatore
        imgSVD = addNoise(img, enableNoise, noiseType, noiseVariance, noiseDensity);
        imgBlu = addNoise(img, enableNoise, noiseType, noiseVariance, noiseDensity);
    end

    % Salvataggio delle immagini processate nelle cartelle di destinazione
    tStep = tic;
    imwrite(imgSVD, fullfile(outputA, fileName));
    imwrite(imgBlu, fullfile(outputB, fileName));
    timeImageSaving(i) = toc(tStep);
end

% Chiusura del timer totale per la generazione
generationBlockTotal = toc(totalGenerationTimer);

%% Estrazione feature

% Si inizializzano le matrici per raccogliere le feature (X) e le 
% label (Y), insieme ai timer per misurare le prestazioni
XA = []; YA = []; XB = []; YB = [];
featureNames = {};
timePreprocessingFeatures = zeros(numImages, 1);
timeFeatureExtractionSVD = zeros(numImages, 1);
timeFeatureExtractionBlu = zeros(numImages, 1);

% Inizializza un array vuoto per salvare gli indici delle feature da rimuovere
idxToRemoveBlu = [];

totalFeatureTimer = tic;
for i = 1:numImages
    tStep = tic;

    % Lettura dell'immagine originale e delle corrispondenti versioni modificate
    original = imread(imds.Files{i});
    original = normalizeImageResolution(original, targetImageSize);
    [~, name, ext] = fileparts(imds.Files{i});
    fileName = [name, ext];

    imgSVD = imread(fullfile(outputA, fileName));
    imgBlu = imread(fullfile(outputB, fileName));
    timePreprocessingFeatures(i) = toc(tStep);

    tStep = tic;
    
    % Calcolo delle feature confrontando l'immagine originale con quella 
    % elaborata tramite SVD, riutilizzando la bounding box precalcolata
    [featuresSVD, featureNames] = computeWatermarkScore(original, imgSVD, bboxList{i});
    
    timeFeatureExtractionSVD(i) = toc(tStep);
    tStep = tic;

    % Calcolo delle feature per l'immagine modificata sul canale Blu.
    featuresBlu = computeWatermarkScore(original, imgBlu, bboxList{i});
    timeFeatureExtractionBlu(i) = toc(tStep);

    % Identifica gli indici da rimuovere solo alla prima iterazione
    if isempty(idxToRemoveBlu)
        idxToRemoveBlu = ismember(featureNames, ...
            {'ssimROI', 'psnrROI', 'diffMeanROI', 'diffStdROI'});
    end

    % Applica il filtro: rimuove le 4 feature specificate da featuresBlu
    featuresBlu(idxToRemoveBlu) = [];

    % Creazione delle label: 1 se l'immagine è stata watermarkata, 0 altrimenti
    label = double(ismember(i, idxSubset));

    % Accumulo dei vettori di feature e delle etichette nelle matrici globali
    XA = [XA; featuresSVD]; %#ok<AGROW>
    YA = [YA; label]; %#ok<AGROW>
    XB = [XB; featuresBlu]; %#ok<AGROW>
    YB = [YB; label]; %#ok<AGROW>
end
featureBlockTotal = toc(totalFeatureTimer);

%% Salvataggio dataset delle feature in CSV

% Creazione della lista dei nomi ridotta per il metodo Blu
featureNamesBlu = featureNames(~idxToRemoveBlu);

% Conversione delle matrici di feature in tabelle
featuresTableA = array2table(XA, 'VariableNames', featureNames);
featuresTableB = array2table(XB, 'VariableNames', featureNamesBlu);

% Aggiunta della label binaria
% 0 = immagine originale
% 1 = immagine watermarkata
featuresTableA.Label = YA;
featuresTableB.Label = YB;

% Aggiunta di una colonna descrittiva della tecnica
featuresTableA.Technique = repmat("SVD", height(featuresTableA), 1);
featuresTableB.Technique = repmat("Blu", height(featuresTableB), 1);

% Riordino colonne: Technique, Label, poi feature
featuresTableA = movevars(featuresTableA, {'Technique','Label'}, 'Before', 1);
featuresTableB = movevars(featuresTableB, {'Technique','Label'}, 'Before', 1);

% Salvataggio CSV separati
writetable(featuresTableA, fullfile(figuresDir, 'features_dataset_SVD.csv'));
writetable(featuresTableB, fullfile(figuresDir, 'features_dataset_Blu.csv'));

% Per poter concatenare le tabelle verticalmente, devono avere le stesse colonne.
% Aggiungiamo le colonne ROI mancanti a featuresTableB impostandole a NaN.
roiFeatureStrings = {'ssimROI', 'psnrROI', 'diffMeanROI', 'diffStdROI'};
for c = 1:length(roiFeatureStrings)
    featuresTableB.(roiFeatureStrings{c}) = NaN(height(featuresTableB), 1);
end

% Salvataggio CSV unico complessivo
featuresTableAll = [featuresTableA; featuresTableB];
writetable(featuresTableAll, fullfile(figuresDir, 'features_dataset_ALL.csv'));

fprintf('\nDataset delle feature salvati in:\n');
fprintf('- %s\n', fullfile(figuresDir, 'features_dataset_SVD.csv'));
fprintf('- %s\n', fullfile(figuresDir, 'features_dataset_Blu.csv'));
fprintf('- %s\n', fullfile(figuresDir, 'features_dataset_ALL.csv'));

fprintf('\nNumero feature estratte SVD: %d\n', numel(featureNames));
fprintf('Numero feature estratte Blu: %d\n', numel(featureNamesBlu));


%% Split condiviso tra i due modelli

% Hold-out split (80-20)
cv = cvpartition(YA, 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx  = test(cv);

globalTestIdx = find(testIdx);

% Lo stesso identico partizionamento viene applicato sia al dataset SVD (A) 
% sia al dataset Blu (B)
XA_train = XA(trainIdx,:);
YA_train = YA(trainIdx);
XA_test  = XA(testIdx,:);
YA_test  = YA(testIdx);

XB_train = XB(trainIdx,:);
YB_train = YB(trainIdx);
XB_test  = XB(testIdx,:);
YB_test  = YB(testIdx);

%% Studio distribuzione classi in train e test

% Questa cella verifica se la distribuzione tra immagini originali
% e watermarkate viene mantenuta correttamente nel train e nel test
%
% Label:
% 0 = immagine originale
% 1 = immagine watermarkata

fprintf('\n============================================================\n');
fprintf('STUDIO DISTRIBUZIONE CLASSI - TRAIN / TEST\n');
fprintf('============================================================\n');

% Controllo coerenza tra label SVD e Blu
if isequal(YA, YB)
    fprintf('\nLe label di Dataset A (SVD) e Dataset B (Blu) coincidono.\n');
else
    warning('Le label di Dataset A (SVD) e Dataset B (Blu) NON coincidono.');
end

% Funzione locale anonima per stampare conteggi e percentuali
printClassDistribution = @(y, name) fprintf([ ...
    '\n%s\n' ...
    'Originali     label 0: %d immagini (%.2f%%)\n' ...
    'Watermarked   label 1: %d immagini (%.2f%%)\n' ...
    'Totale               : %d immagini\n'], ...
    name, ...
    sum(y == 0), 100 * sum(y == 0) / numel(y), ...
    sum(y == 1), 100 * sum(y == 1) / numel(y), ...
    numel(y));

% Distribuzione totale
printClassDistribution(YA, 'Dataset totale');

% Distribuzione train
printClassDistribution(YA(trainIdx), 'Training set');

% Distribuzione test
printClassDistribution(YA(testIdx), 'Test set');

% Tabella riepilogativa
classSplitSummary = table( ...
    ["Totale"; "Train"; "Test"], ...
    [sum(YA == 0); sum(YA(trainIdx) == 0); sum(YA(testIdx) == 0)], ...
    [sum(YA == 1); sum(YA(trainIdx) == 1); sum(YA(testIdx) == 1)], ...
    [100 * sum(YA == 0) / numel(YA); ...
     100 * sum(YA(trainIdx) == 0) / numel(YA(trainIdx)); ...
     100 * sum(YA(testIdx) == 0) / numel(YA(testIdx))], ...
    [100 * sum(YA == 1) / numel(YA); ...
     100 * sum(YA(trainIdx) == 1) / numel(YA(trainIdx)); ...
     100 * sum(YA(testIdx) == 1) / numel(YA(testIdx))], ...
    'VariableNames', {'Split', 'NumOriginal', 'NumWatermarked', ...
                      'PercOriginal', 'PercWatermarked'} );

disp(' ');
disp('Tabella riepilogativa distribuzione classi:');
disp(classSplitSummary);

% Salvataggio del riepilogo nella cartella delle figure
writetable(classSplitSummary, fullfile(figuresDir, 'class_distribution_train_test.csv'));

fprintf('\nRiepilogo distribuzione classi salvato in:\n%s\n', ...
    fullfile(figuresDir, 'class_distribution_train_test.csv'));

% Controllo warning se nel test ci sono pochi watermark
numWatermarkedTest = sum(YA(testIdx) == 1);

if numWatermarkedTest == 0
    warning(['Nel test set non ci sono immagini watermarkate. ', ...
             'Le metriche Precision, Recall e F1 non sono affidabili.']);
elseif numWatermarkedTest < 5
    warning(['Nel test set ci sono solo %d immagini watermarkate. ', ...
             'Le metriche possono essere instabili.'], numWatermarkedTest);
else
    fprintf('\nIl test set contiene %d immagini watermarkate.\n', numWatermarkedTest);
end

fprintf('============================================================\n');
%% Normalizzazione z-score separata per ciascun modello

% Calcolo di media e deviazione standard esclusivamente sul training set
muA = mean(XA_train, 1);
sigmaA = std(XA_train, 0, 1);

% Protezione contro deviazioni standard nulle o valori non validi
sigmaA(sigmaA < 1e-12) = 1;
muA(~isfinite(muA)) = 0;
sigmaA(~isfinite(sigmaA)) = 1;

% Applicazione della normalizzazione Z-score sia ai dati di train che 
% di test per SVD
XA_train_norm = (XA_train - muA) ./ sigmaA;
XA_test_norm  = (XA_test  - muA) ./ sigmaA;

XA_train_norm(~isfinite(XA_train_norm)) = 0;
XA_test_norm(~isfinite(XA_test_norm)) = 0;

% Stesso procedimento per le feature del metodo Blu
muB = mean(XB_train, 1);
sigmaB = std(XB_train, 0, 1);
sigmaB(sigmaB < 1e-12) = 1;
muB(~isfinite(muB)) = 0;
sigmaB(~isfinite(sigmaB)) = 1;

XB_train_norm = (XB_train - muB) ./ sigmaB;
XB_test_norm  = (XB_test  - muB) ./ sigmaB;

XB_train_norm(~isfinite(XB_train_norm)) = 0;
XB_test_norm(~isfinite(XB_test_norm)) = 0;


%% Hyperparameter tuning SVM e Random Forest

% Si ottimizzano i parametri dei modelli per trovare la configurazione più 
% performante.
% Obiettivo di selezione: F1-score; in caso di parita' viene considerata l'AUC.
enableHyperparameterTuning = true;

if enableHyperparameterTuning
    fprintf('\nAvvio hyperparameter tuning SVM/RF...\n');

    % Ricerca dei parametri ottimali per le SVM
    [bestSVMParamsA, svmTuningResultsA, tuningTimeSVMA] = tuneSVMHyperparameters( ...
        XA_train_norm, YA_train, 'SVD', figuresDir);
    [bestSVMParamsB, svmTuningResultsB, tuningTimeSVMB] = tuneSVMHyperparameters( ...
        XB_train_norm, YB_train, 'Blu', figuresDir);

    % Ricerca dei parametri ottimali per i modelli Random Forest
    [bestRFParamsA, rfTuningResultsA, tuningTimeRFA] = tuneRFHyperparameters( ...
        XA_train_norm, YA_train, 'SVD', figuresDir);
    [bestRFParamsB, rfTuningResultsB, tuningTimeRFB] = tuneRFHyperparameters( ...
        XB_train_norm, YB_train, 'Blu', figuresDir);

    % Generazione dei grafici per visualizzare l'andamento del tuning
    plotHyperparameterTuningResults( ...
        svmTuningResultsA, svmTuningResultsB, rfTuningResultsA, rfTuningResultsB, figuresDir);

    % Stampa a video dei migliori parametri trovati
    fprintf('Miglior SVM SVD: C = %.3g, KernelScale = %.3g, F1 val = %.3f\n', ...
        bestSVMParamsA.BoxConstraint, bestSVMParamsA.KernelScale, bestSVMParamsA.ObjectiveF1);
    fprintf('Miglior SVM Blu: C = %.3g, KernelScale = %.3g, F1 val = %.3f\n', ...
        bestSVMParamsB.BoxConstraint, bestSVMParamsB.KernelScale, bestSVMParamsB.ObjectiveF1);
    fprintf('Miglior RF SVD: NumTrees = %d, MaxNumSplits = %d, F1 val = %.3f\n', ...
        bestRFParamsA.NumLearningCycles, bestRFParamsA.MaxNumSplits, bestRFParamsA.ObjectiveF1);
    fprintf('Miglior RF Blu: NumTrees = %d, MaxNumSplits = %d, F1 val = %.3f\n', ...
        bestRFParamsB.NumLearningCycles, bestRFParamsB.MaxNumSplits, bestRFParamsB.ObjectiveF1);
else
    % Se il tuning è disabilitato, si assegnano valori di default
    bestSVMParamsA = struct('BoxConstraint', 1, 'KernelScale', 1, 'ObjectiveF1', NaN, 'ObjectiveAUC', NaN);
    bestSVMParamsB = struct('BoxConstraint', 1, 'KernelScale', 1, 'ObjectiveF1', NaN, 'ObjectiveAUC', NaN);
    bestRFParamsA = struct('NumLearningCycles', 100, 'MaxNumSplits', 10, 'ObjectiveF1', NaN, 'ObjectiveAUC', NaN);
    bestRFParamsB = struct('NumLearningCycles', 100, 'MaxNumSplits', 10, 'ObjectiveF1', NaN, 'ObjectiveAUC', NaN);
    tuningTimeSVMA = 0;
    tuningTimeSVMB = 0;
    tuningTimeRFA = 0;
    tuningTimeRFB = 0;
end

%% Logistic Regression

tStep = tic;

% Addestramento dei modelli di Regressione Logistica
modelA = fitclinear(XA_train_norm, YA_train, 'Learner', 'logistic');
trainTimeLogRegA = toc(tStep);
tStep = tic;
modelB = fitclinear(XB_train_norm, YB_train, 'Learner', 'logistic');
trainTimeLogRegB = toc(tStep);

tStep = tic;

% Generazione delle predizioni sul test set per entrambi i modelli
YA_pred = predict(modelA, XA_test_norm);
predictTimeLogRegA = toc(tStep);
tStep = tic;
YB_pred = predict(modelB, XB_test_norm);
predictTimeLogRegB = toc(tStep);

%% Metriche Logistic Regression

% Calcolo delle metriche di entrambi i modelli di Regressione
% Logistica (Accuratezza, Precision, Recall, F1)
metricsA = metrics(YA_test, YA_pred);
metricsB = metrics(YB_test, YB_pred);

% Stampa dei risultati per il modello SVD
fprintf('\nLogistic Regression - Model A (SVD):\n');
fprintf('Accuracy  : %.3f\n', metricsA.Accuracy);
fprintf('Precision : %.3f\n', metricsA.Precision);
fprintf('Recall    : %.3f\n', metricsA.Recall);
fprintf('F1-score  : %.3f\n', metricsA.F1);

% Generazione della matrice di confusione (SVD)
f = figure;
confusionchart(metricsA.ConfusionMatrix, {'Original', 'Watermarked'});
title('Confusion Matrix - Logistic Regression - WatermarkSVD');
saveas(f, fullfile(figuresDir, 'cm_logreg_svd.png'));

% Stampa dei risultati per il modello Blu
fprintf('\nLogistic Regression - Model B (Blu):\n');
fprintf('Accuracy  : %.3f\n', metricsB.Accuracy);
fprintf('Precision : %.3f\n', metricsB.Precision);
fprintf('Recall    : %.3f\n', metricsB.Recall);
fprintf('F1-score  : %.3f\n', metricsB.F1);

% Generazione della matrice di confusione (Blu)
f = figure;
confusionchart(metricsB.ConfusionMatrix, {'Original', 'Watermarked'});
title('Confusion Matrix - Logistic Regression - WatermarkBlu');
saveas(f, fullfile(figuresDir, 'cm_logreg_blu.png'));

%% ROC Curve Logistic Regression

% Calcolo degli score per la generazione della curva ROC
[~, scoresA] = predict(modelA, XA_test_norm);
[~, scoresB] = predict(modelB, XB_test_norm);

% Selezione degli score relativi alla classe positiva
if size(scoresA,2) > 1; scoresA = scoresA(:,2); end
if size(scoresB,2) > 1; scoresB = scoresB(:,2); end

% Calcolo dei tassi di FP e TP per il tracciamento della curva
[Xa, Ya, ~, aucA] = perfcurve(YA_test, scoresA, 1);
[Xb, Yb, ~, aucB] = perfcurve(YB_test, scoresB, 1);

% Visualizzazione grafica delle curve ROC
figure;
plot(Xa, Ya, 'b-', 'LineWidth', 2); hold on;
plot(Xb, Yb, 'r--', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - Logistic Regression');
legend(sprintf('WatermarkSVD (AUC = %.5f)', aucA), ...
       sprintf('WatermarkBlu (AUC = %.5f)', aucB), ...
       'Location', 'SouthEast');
grid on;
saveas(gcf, fullfile(figuresDir, 'roc_logreg.png'));


%% Support Vector Machine

% Addestramento di un modello non lineare SVM con kernel RBF (Radial Basis 
% Function). Vengono utilizzati gli iperparametri ottimali (KernelScale e 
% BoxConstraint) trovati in precedenza.
tStep = tic;
modelSVM_A = fitcsvm(XA_train_norm, YA_train, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', bestSVMParamsA.KernelScale, ...
    'BoxConstraint', bestSVMParamsA.BoxConstraint, ...
    'Standardize', false, ...
    'ClassNames', [0 1]);

% Calibrazione opzionale delle posterior probabilities, utile per ROC/AUC.
% Se la calibrazione non va a buon fine, vengono usati gli score grezzi.
try
    modelSVM_A = fitPosterior(modelSVM_A, XA_train_norm, YA_train);
catch ME
    warning(['fitPosterior SVM A non disponibile o non riuscito: %s.' ...
        ' Uso degli score SVM grezzi.'], ME.message);
end
trainTimeSVMA = toc(tStep);

% Si ripete l'addestramento e la calibrazione per il modello basato sul canale Blu
tStep = tic;
modelSVM_B = fitcsvm(XB_train_norm, YB_train, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', bestSVMParamsB.KernelScale, ...
    'BoxConstraint', bestSVMParamsB.BoxConstraint, ...
    'Standardize', false, ...
    'ClassNames', [0 1]);
try
    modelSVM_B = fitPosterior(modelSVM_B, XB_train_norm, YB_train);
catch ME
    warning(['fitPosterior SVM B non disponibile o non riuscito: %s.' ...
        ' Uso degli score SVM grezzi.'], ME.message);
end
trainTimeSVMB = toc(tStep);

tStep = tic;

% Effettua le predizioni sui rispettivi test set per valutarne le performance
YA_pred_SVM = predict(modelSVM_A, XA_test_norm);
predictTimeSVMA = toc(tStep);

tStep = tic;
YB_pred_SVM = predict(modelSVM_B, XB_test_norm);
predictTimeSVMB = toc(tStep);

%% Metriche Support Vector Machine

% Calcolo delle metriche di entrambi i modelli SVM
metricsSVM_A = metrics(YA_test, YA_pred_SVM);
metricsSVM_B = metrics(YB_test, YB_pred_SVM);

% Stampa dei risultati per il modello SVD
fprintf('\nSVM - Model A (SVD):\n');
fprintf('Accuracy  : %.3f\n', metricsSVM_A.Accuracy);
fprintf('Precision : %.3f\n', metricsSVM_A.Precision);
fprintf('Recall    : %.3f\n', metricsSVM_A.Recall);
fprintf('F1-score  : %.3f\n', metricsSVM_A.F1);

% Generazione della matrice di confusione (SVD)
f = figure;
confusionchart(metricsSVM_A.ConfusionMatrix, {'Original', 'Watermarked'});
title('Confusion Matrix - SVM - WatermarkSVD');
saveas(f, fullfile(figuresDir, 'cm_svm_svd.png'));

% Stampa dei risultati per il modello Blu
fprintf('\nSVM - Model B (Blu):\n');
fprintf('Accuracy  : %.3f\n', metricsSVM_B.Accuracy);
fprintf('Precision : %.3f\n', metricsSVM_B.Precision);
fprintf('Recall    : %.3f\n', metricsSVM_B.Recall);
fprintf('F1-score  : %.3f\n', metricsSVM_B.F1);

% Generazione della matrice di confusione (Blu)
f = figure;
confusionchart(metricsSVM_B.ConfusionMatrix, {'Original', 'Watermarked'});
title('Confusion Matrix - SVM - WatermarkBlu');
saveas(f, fullfile(figuresDir, 'cm_svm_blu.png'));

%% ROC Curve Support Vector Machine

% Calcolo degli score per la generazione della curva ROC
[~, scoresSVM_A] = predict(modelSVM_A, XA_test_norm);
[~, scoresSVM_B] = predict(modelSVM_B, XB_test_norm);

% Selezione degli score relativi alla classe positiva
scoresSVM_A = extractPositiveClassScores(scoresSVM_A);
scoresSVM_B = extractPositiveClassScores(scoresSVM_B);

% Calcolo dei tassi di FP e TP per il tracciamento della curva
[Xa_SVM, Ya_SVM, ~, aucSVM_A] = perfcurve(YA_test, scoresSVM_A, 1);
[Xb_SVM, Yb_SVM, ~, aucSVM_B] = perfcurve(YB_test, scoresSVM_B, 1);

% Visualizzazione grafica delle curve ROC
figure;
plot(Xa_SVM, Ya_SVM, 'b-', 'LineWidth', 2); hold on;
plot(Xb_SVM, Yb_SVM, 'r--', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - SVM');
legend(sprintf('WatermarkSVD (AUC = %.5f)', aucSVM_A), ...
       sprintf('WatermarkBlu (AUC = %.5f)', aucSVM_B), ...
       'Location', 'SouthEast');
grid on;
saveas(gcf, fullfile(figuresDir, 'roc_svm.png'));


%% Random Forest

% Addestramento di un modello d'insieme (Random Forest) mediante il metodo Bagging.
% Si utilizzano il numero di alberi (NumLearningCycles) e gli split massimi (MaxNumSplits)
% calcolati precedentemente nella fase di hyperparameter tuning.

tStep = tic;
modelRF_A = fitcensemble(XA_train_norm, YA_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', bestRFParamsA.NumLearningCycles, ...
    'Learners', templateTree('MaxNumSplits', bestRFParamsA.MaxNumSplits), ...
    'ClassNames', [0 1]);
trainTimeRFA = toc(tStep);

tStep = tic;
modelRF_B = fitcensemble(XB_train_norm, YB_train, ...
    'Method', 'Bag', ...
    'NumLearningCycles', bestRFParamsB.NumLearningCycles, ...
    'Learners', templateTree('MaxNumSplits', bestRFParamsB.MaxNumSplits), ...
    'ClassNames', [0 1]);
trainTimeRFB = toc(tStep);

% Calcolo delle predizioni per il test set al fine di ricavare le metriche
tStep = tic;
YA_pred_RF = predict(modelRF_A, XA_test_norm);
predictTimeRFA = toc(tStep);

tStep = tic;
YB_pred_RF = predict(modelRF_B, XB_test_norm);
predictTimeRFB = toc(tStep);

%% Metriche Random Forest

% Calcolo delle metriche di entrambi i modelli Random Forest
metricsRF_A = metrics(YA_test, YA_pred_RF);
metricsRF_B = metrics(YB_test, YB_pred_RF);

% Stampa dei risultati per il modello SVD
fprintf('\nRandom Forest - Model A (SVD):\n');
fprintf('Accuracy  : %.3f\n', metricsRF_A.Accuracy);
fprintf('Precision : %.3f\n', metricsRF_A.Precision);
fprintf('Recall    : %.3f\n', metricsRF_A.Recall);
fprintf('F1-score  : %.3f\n', metricsRF_A.F1);

% Generazione della matrice di confusione (SVD)
f = figure;
confusionchart(metricsRF_A.ConfusionMatrix, {'Original', 'Watermarked'});
title('Confusion Matrix - Random Forest - WatermarkSVD');
saveas(f, fullfile(figuresDir, 'cm_rf_svd.png'));

% Stampa dei risultati per il modello Blu
fprintf('\nRandom Forest - Model B (Blu):\n');
fprintf('Accuracy  : %.3f\n', metricsRF_B.Accuracy);
fprintf('Precision : %.3f\n', metricsRF_B.Precision);
fprintf('Recall    : %.3f\n', metricsRF_B.Recall);
fprintf('F1-score  : %.3f\n', metricsRF_B.F1);

% Generazione della matrice di confusione (Blu)
f = figure;
confusionchart(metricsRF_B.ConfusionMatrix, {'Original', 'Watermarked'});
title('Confusion Matrix - Random Forest - WatermarkBlu');
saveas(f, fullfile(figuresDir, 'cm_rf_blu.png'));

%% ROC Curve Random Forest

% Calcolo degli score per la generazione della curva ROC
[~, scoresRF_A] = predict(modelRF_A, XA_test_norm);
[~, scoresRF_B] = predict(modelRF_B, XB_test_norm);

% Selezione degli score relativi alla classe positiva
if size(scoresRF_A,2) > 1; scoresRF_A = scoresRF_A(:,2); end
if size(scoresRF_B,2) > 1; scoresRF_B = scoresRF_B(:,2); end

% Calcolo dei tassi di FP e TP per il tracciamento della curva
[Xa_RF, Ya_RF, ~, aucRF_A] = perfcurve(YA_test, scoresRF_A, 1);
[Xb_RF, Yb_RF, ~, aucRF_B] = perfcurve(YB_test, scoresRF_B, 1);

% Visualizzazione grafica delle curve ROC
figure;
plot(Xa_RF, Ya_RF, 'b-', 'LineWidth', 2); hold on;
plot(Xb_RF, Yb_RF, 'r--', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - Random Forest');
legend(sprintf('WatermarkSVD (AUC = %.5f)', aucRF_A), ...
       sprintf('WatermarkBlu (AUC = %.5f)', aucRF_B), ...
       'Location', 'SouthEast');
grid on;
saveas(gcf, fullfile(figuresDir, 'roc_rf.png'));


%% Riepilogo comparativo dei modelli

% Generazione di una tabella riassuntiva per confrontare i tre modelli
resultsSummary = table( ...
    {'Logistic Regression'; 'SVM'; 'Random Forest'}, ...
    [metricsA.Accuracy; metricsSVM_A.Accuracy; metricsRF_A.Accuracy], ...
    [metricsA.Precision; metricsSVM_A.Precision; metricsRF_A.Precision], ...
    [metricsA.Recall; metricsSVM_A.Recall; metricsRF_A.Recall], ...
    [metricsA.F1; metricsSVM_A.F1; metricsRF_A.F1], ...
    [aucA; aucSVM_A; aucRF_A], ...
    [metricsB.Accuracy; metricsSVM_B.Accuracy; metricsRF_B.Accuracy], ...
    [metricsB.Precision; metricsSVM_B.Precision; metricsRF_B.Precision], ...
    [metricsB.Recall; metricsSVM_B.Recall; metricsRF_B.Recall], ...
    [metricsB.F1; metricsSVM_B.F1; metricsRF_B.F1], ...
    [aucB; aucSVM_B; aucRF_B], ...
    'VariableNames', {'Model', ...
    'SVD_Accuracy', 'SVD_Precision', 'SVD_Recall', 'SVD_F1', 'SVD_AUC', ...
    'Blu_Accuracy', 'Blu_Precision', 'Blu_Recall', 'Blu_F1', 'Blu_AUC'});

% Stampa a video della tabella riassuntiva dei modelli
disp('Riepilogo comparativo dei modelli:');
disp(resultsSummary);

% Salvataggio della tabella comparativa del trial corrente
writetable(resultsSummary, fullfile(figuresDir, 'results_summary.csv'));

% Conversione del riepilogo in formato long, utile per aggregare automaticamente
% i risultati di più trial e generare i grafici F1/AUC al variare di idx e alpha.
trialMetricsLong = makeTrialMetricsLongTable( ...
    trialNumber, watermarkRatio, alphaSVD, alphaBlue, resultsSummary);

writetable(trialMetricsLong, fullfile(figuresDir, 'trial_metrics_long.csv'));

% Crea una tabella per riepilogare i risultati dell'hyperparameter tuning.
% Vengono registrati i parametri ottimizzati, le migliori configurazioni trovate,
% le metriche di validazione (F1 e AUC) e il tempo impiegato per la ricerca
hyperparameterSummary = table( ...
    {'SVM SVD'; 'SVM Blu'; 'Random Forest SVD'; 'Random Forest Blu'}, ...
    {'BoxConstraint, KernelScale'; 'BoxConstraint, KernelScale'; ...
     'NumLearningCycles, MaxNumSplits'; 'NumLearningCycles, MaxNumSplits'}, ...
    {sprintf('C=%.3g, KernelScale=%.3g', bestSVMParamsA.BoxConstraint, bestSVMParamsA.KernelScale); ...
     sprintf('C=%.3g, KernelScale=%.3g', bestSVMParamsB.BoxConstraint, bestSVMParamsB.KernelScale); ...
     sprintf('NumTrees=%d, MaxNumSplits=%d', bestRFParamsA.NumLearningCycles, bestRFParamsA.MaxNumSplits); ...
     sprintf('NumTrees=%d, MaxNumSplits=%d', bestRFParamsB.NumLearningCycles, bestRFParamsB.MaxNumSplits)}, ...
    [bestSVMParamsA.ObjectiveF1; bestSVMParamsB.ObjectiveF1; ...
     bestRFParamsA.ObjectiveF1; bestRFParamsB.ObjectiveF1], ...
    [bestSVMParamsA.ObjectiveAUC; bestSVMParamsB.ObjectiveAUC; ...
     bestRFParamsA.ObjectiveAUC; bestRFParamsB.ObjectiveAUC], ...
    [tuningTimeSVMA; tuningTimeSVMB; tuningTimeRFA; tuningTimeRFB], ...
    'VariableNames', {'Model', 'TunedParameters', 'BestConfiguration', ...
    'ValidationF1', 'ValidationAUC', 'TuningTime_s'});

% Stampa a video del riepilogo del tuning
disp('Riepilogo hyperparameter tuning:');
disp(hyperparameterSummary);
writetable(hyperparameterSummary, fullfile(figuresDir, 'hyperparameter_tuning_summary.csv'));


%% Analisi della complessita' computazionale

% Estrazione del numero di immagini di test e formattazione della risoluzione
numTestImages = numel(YA_test);
imageSizeLabel = sprintf('%dx%d', targetImageSize(1), targetImageSize(2));

% Somma dei tempi parziali calcolati all'interno dei cicli for
preprocessGenerationTotal = sum(timePreprocessingGeneration);
preprocessFeaturesTotal = sum(timePreprocessingFeatures);
yoloDetectionTotal = sum(timeYOLODetection);
imageSavingTotal = sum(timeImageSaving);
watermarkSVDTotal = sum(timeWatermarkSVD);
watermarkBluTotal = sum(timeWatermarkBlu);
featureSVDTotal = sum(timeFeatureExtractionSVD);
featureBluTotal = sum(timeFeatureExtractionBlu);

% Calcolo del numero medio di bounding box rilevate per immagine.
meanDetectionsPerImage = mean(numDetections);

% Somma dei tempi impiegati per l'hyperparameter tuning
hpoSVDTotal = tuningTimeSVMA + tuningTimeRFA;
hpoBluTotal = tuningTimeSVMB + tuningTimeRFB;

% Calcolo della dimensione su disco dei modelli addestrati
modelSizeLogRegA = getModelSize(modelA, 'model_logreg_svd');
modelSizeLogRegB = getModelSize(modelB, 'model_logreg_blu');
modelSizeSVMA = getModelSize(modelSVM_A, 'model_svm_svd');
modelSizeSVMB = getModelSize(modelSVM_B, 'model_svm_blu');
modelSizeRFA = getModelSize(modelRF_A, 'model_rf_svd');
modelSizeRFB = getModelSize(modelRF_B, 'model_rf_blu');

% Crea una tabella riassuntiva che raggruppa tutte le metriche
% prestazionali (Accuracy, F1, AUC) e i tempi operativi per ogni 
% combinazione Metodo + Classificatore
complexitySummary = table( ...
    {'SVD + Logistic Regression'; 'SVD + SVM'; 'SVD + Random Forest'; ...
     'Blu + Logistic Regression'; 'Blu + SVM'; 'Blu + Random Forest'}, ...
    {'SVD luminanza'; 'SVD luminanza'; 'SVD luminanza'; ...
     'Canale blu'; 'Canale blu'; 'Canale blu'}, ...
    {'Logistic Regression'; 'SVM'; 'Random Forest'; ...
     'Logistic Regression'; 'SVM'; 'Random Forest'}, ...
    repmat({imageSizeLabel}, 6, 1), ...
    repmat({detectorType}, 6, 1), ...
    repmat(numImages, 6, 1), ...
    repmat(numTestImages, 6, 1), ...
    repmat(detectorInitTime, 6, 1), ...
    repmat(generationBlockTotal, 6, 1), ...
    repmat(featureBlockTotal, 6, 1), ...
    [hpoSVDTotal; hpoSVDTotal; hpoSVDTotal; hpoBluTotal; hpoBluTotal; hpoBluTotal], ...
    repmat(preprocessGenerationTotal, 6, 1), ...
    repmat(preprocessFeaturesTotal, 6, 1), ...
    repmat(yoloDetectionTotal, 6, 1), ...
    [watermarkSVDTotal; watermarkSVDTotal; watermarkSVDTotal; ...
     watermarkBluTotal; watermarkBluTotal; watermarkBluTotal], ...
    [featureSVDTotal; featureSVDTotal; featureSVDTotal; ...
     featureBluTotal; featureBluTotal; featureBluTotal], ...
    repmat(imageSavingTotal, 6, 1), ...
    [trainTimeLogRegA; trainTimeSVMA; trainTimeRFA; ...
     trainTimeLogRegB; trainTimeSVMB; trainTimeRFB], ...
    [predictTimeLogRegA; predictTimeSVMA; predictTimeRFA; ...
     predictTimeLogRegB; predictTimeSVMB; predictTimeRFB], ...
    1000 * [predictTimeLogRegA; predictTimeSVMA; predictTimeRFA; ...
            predictTimeLogRegB; predictTimeSVMB; predictTimeRFB] / max(numTestImages, 1), ...
    1000 * ([preprocessGenerationTotal; preprocessGenerationTotal; preprocessGenerationTotal; ...
             preprocessGenerationTotal; preprocessGenerationTotal; preprocessGenerationTotal] + ...
            [yoloDetectionTotal; yoloDetectionTotal; yoloDetectionTotal; ...
             yoloDetectionTotal; yoloDetectionTotal; yoloDetectionTotal] + ...
            [watermarkSVDTotal; watermarkSVDTotal; watermarkSVDTotal; ...
             watermarkBluTotal; watermarkBluTotal; watermarkBluTotal] + ...
            [featureSVDTotal; featureSVDTotal; featureSVDTotal; ...
             featureBluTotal; featureBluTotal; featureBluTotal] + ...
            [predictTimeLogRegA; predictTimeSVMA; predictTimeRFA; ...
             predictTimeLogRegB; predictTimeSVMB; predictTimeRFB]) / max(numImages, 1), ...
    [modelSizeLogRegA; modelSizeSVMA; modelSizeRFA; ...
     modelSizeLogRegB; modelSizeSVMB; modelSizeRFB], ...
    repmat(meanDetectionsPerImage, 6, 1), ...
    [metricsA.Accuracy; metricsSVM_A.Accuracy; metricsRF_A.Accuracy; ...
     metricsB.Accuracy; metricsSVM_B.Accuracy; metricsRF_B.Accuracy], ...
    [metricsA.F1; metricsSVM_A.F1; metricsRF_A.F1; ...
     metricsB.F1; metricsSVM_B.F1; metricsRF_B.F1], ...
    [aucA; aucSVM_A; aucRF_A; aucB; aucSVM_B; aucRF_B], ...
    'VariableNames', {'Pipeline', 'WatermarkMethod', 'Classifier', ...
    'ImageSize', 'DetectorType', 'NumImages', 'NumTestImages', ...
    'DetectorInitTime_s', 'DatasetGenerationBlockTotal_s', ...
    'FeatureExtractionBlockTotal_s', 'HyperparameterTuningTime_s', ...
    'PreprocessingGenerationTotal_s', ...
    'PreprocessingFeatureTotal_s', 'YOLODetectionTotal_s', ...
    'WatermarkTotal_s', 'FeatureExtractionTotal_s', 'ImageSavingTotal_s', ...
    'TrainingTime_s', 'PredictionTime_s', 'AvgPredictionTimePerImage_ms', ...
    'AvgPipelineTimePerImage_ms', 'ModelSizeMB', 'MeanDetectionsPerImage', ...
    'Accuracy', 'F1', 'AUC'});

% Stampa a video della tabella riassuntiva
disp('Riepilogo della complessita'' computazionale:');
disp(complexitySummary);
writetable(complexitySummary, fullfile(figuresDir, 'computational_complexity_summary.csv'));

% Crea una tabella per un'analisi dettagliata dei tempi 
% impiegati dai singoli blocchi logici del codice
complexityBlocks = table( ...
    {'Creazione detector'; 'Generazione dataset A/B'; 'Estrazione feature'; ...
     'Hyperparameter tuning SVM SVD'; 'Hyperparameter tuning SVM Blu'; ...
     'Hyperparameter tuning Random Forest SVD'; 'Hyperparameter tuning Random Forest Blu'; ...
     'Training Logistic Regression SVD'; 'Training SVM SVD'; 'Training Random Forest SVD'; ...
     'Training Logistic Regression Blu'; 'Training SVM Blu'; 'Training Random Forest Blu'; ...
     'Predizione Logistic Regression SVD'; 'Predizione SVM SVD'; 'Predizione Random Forest SVD'; ...
     'Predizione Logistic Regression Blu'; 'Predizione SVM Blu'; 'Predizione Random Forest Blu'}, ...
    [detectorInitTime; generationBlockTotal; featureBlockTotal; ...
     tuningTimeSVMA; tuningTimeSVMB; tuningTimeRFA; tuningTimeRFB; ...
     trainTimeLogRegA; trainTimeSVMA; trainTimeRFA; ...
     trainTimeLogRegB; trainTimeSVMB; trainTimeRFB; ...
     predictTimeLogRegA; predictTimeSVMA; predictTimeRFA; ...
     predictTimeLogRegB; predictTimeSVMB; predictTimeRFB], ...
    'VariableNames', {'Block', 'Time_s'});

% Salvataggio della tabella dei tempi per blocchi logici
writetable(complexityBlocks, fullfile(figuresDir, 'computational_complexity_blocks.csv'));

% Generazione di un grafico a barre della complessità computazionale
createComplexityBarPlot(complexitySummary, figuresDir);


%% Generazione di grafici extra
% 1) Boxplot di feature chiave per classe
% 2) Feature importance della Random Forest
% 3) Esempi visivi: originale, watermarked e heatmap differenze

generatePlots( ...
    XA_test, YA_test, ...
    XB_test, YB_test, ...
    featureNames, modelRF_A, modelRF_B, ...
    imds, outputA, outputB, bboxList, globalTestIdx, figuresDir, targetImageSize);

% Conferma a schermo del completamento del salvataggio
fprintf('\nFigure salvate nella cartella: %s\n', figuresDir);

% In modalità sweep, lo script runSweepAlphaIdx.m leggerà il file
% trial_metrics_long.csv appena salvato e costruirà la tabella aggregata.
