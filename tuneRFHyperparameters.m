function [bestParams, tuningResults, tuningTime] = tuneRFHyperparameters(XTrain, YTrain, datasetLabel, figuresDir)
% Esegue tuning manuale della Random Forest

    % Se la cartella per il salvataggio dei grafici non viene fornita,
    % si utilizza la cartella di lavoro corrente
    if nargin < 4 || isempty(figuresDir)
        figuresDir = pwd;
    end
    
    % Avvio cronometro per misurare il tempo totale impiegato dal processo di tuning
    timerTuning = tic;
    
    % Definisce la griglia dei parametri da esplorare
    numTreesValues = [50 100 200];
    maxSplitsValues = [5 10 20];
    
    % Effettua uno split interno sui dati di training
    cvTune = cvpartition(YTrain, 'HoldOut', 0.25);
    idxTr = training(cvTune);
    idxVal = test(cvTune);

    % Estrae i sottoinsiemi di training e validazione interni
    Xtr = XTrain(idxTr, :);
    Ytr = YTrain(idxTr);
    Xval = XTrain(idxVal, :);
    Yval = YTrain(idxVal);

    % Prealloca le variabili per memorizzare i risultati di ogni iterazione 
    % e tenere traccia della configurazione più performante
    rows = {};
    bestScore = -inf;
    bestParams = struct('NumLearningCycles', 100, 'MaxNumSplits', 10, ...
    'ObjectiveF1', NaN, 'ObjectiveAUC', NaN);

    % Grid Search iterando su tutte le combinazioni possibili
    for i = 1:numel(numTreesValues)
        for j = 1:numel(maxSplitsValues)
            nTrees = numTreesValues(i);
            maxSplits = maxSplitsValues(j);

            % Si crea un template per i singoli alberi decisionali 
            % impostandone la profondità
            treeTemplate = templateTree('MaxNumSplits', maxSplits);

            % Metodo Bagging con i parametri correnti
            tStep = tic;
            mdl = fitcensemble(Xtr, Ytr, ...
                'Method', 'Bag', ...
                'NumLearningCycles', nTrees, ...
                'Learners', treeTemplate, ...
                'ClassNames', [0 1]);
            trainTime = toc(tStep);

            % Calcolo delle predizioni e degli score sul validation set
            tStep = tic;
            [predVal, scoreVal] = predict(mdl, Xval);
            predictionTime = toc(tStep);

            % Calcolo delle metriche di classificazione
            m = metrics(Yval, predVal);

            % Calcolo dell'Area Sotto la Curva (AUC)
            aucVal = NaN;
            try
                scorePositive = extractPositiveClassScores(scoreVal);
                [~, ~, ~, aucVal] = perfcurve(Yval, scorePositive, 1);
            catch
                aucVal = NaN;
            end

            % Si accoda una nuova riga con tutti i risultati della 
            % configurazione corrente
            rows(end+1, :) = {datasetLabel, nTrees, maxSplits, ...
                m.Accuracy, m.Precision, m.Recall, m.F1, aucVal, ...
                trainTime, predictionTime}; %#ok<AGROW>

            % Obiettivo principale: F1. In caso di parita', usa AUC.
            objective = m.F1;
            if objective > bestScore || (abs(objective - bestScore) ...
                < 1e-12 && aucVal > bestParams.ObjectiveAUC)
                bestScore = objective;
                bestParams.NumLearningCycles = nTrees;
                bestParams.MaxNumSplits = maxSplits;
                bestParams.ObjectiveF1 = m.F1;
                bestParams.ObjectiveAUC = aucVal;
            end
        end
    end

    % Converte l'array di celle in una tabella per analisi ed esportazione
    tuningResults = cell2table(rows, 'VariableNames', { ...
        'Dataset', 'NumLearningCycles', 'MaxNumSplits', 'Accuracy', 'Precision', ...
        'Recall', 'F1', 'AUC', 'TrainingTime_s', 'PredictionTime_s'});

    % Ferma il cronometro principale
    tuningTime = toc(timerTuning);

    % Salva la tabella riassuntiva del tuning su disco in formato CSV
    writetable(tuningResults, fullfile(figuresDir, ...
    sprintf('hpo_rf_%s.csv', lower(datasetLabel))));
end
