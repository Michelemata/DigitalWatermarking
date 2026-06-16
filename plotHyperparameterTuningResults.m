function plotHyperparameterTuningResults(svmResultsA, svmResultsB, rfResultsA, rfResultsB, figuresDir)
% Salva grafici del tuning SVM/RF.

    % Se la directory di output non è fornita, si utilizza la cartella corrente (pwd)
    if nargin < 5 || isempty(figuresDir)
        figuresDir = pwd;
    end
    
    % Generazione delle heatmap per i modelli SVM (metodo SVD e metodo Blu)
    plotSVMHeatmap(svmResultsA, fullfile(figuresDir, 'hpo_svm_svd_f1_heatmap.png'), 'SVM tuning - SVD luminanza');
    plotSVMHeatmap(svmResultsB, fullfile(figuresDir, 'hpo_svm_blu_f1_heatmap.png'), 'SVM tuning - Watermark blu');

    % Generazione dei grafici a linee per i modelli Random Forest
    plotRFLinePlot(rfResultsA, fullfile(figuresDir, 'hpo_rf_svd_f1_lines.png'), 'Random Forest tuning - SVD luminanza');
    plotRFLinePlot(rfResultsB, fullfile(figuresDir, 'hpo_rf_blu_f1_lines.png'), 'Random Forest tuning - Watermark blu');
end

%% FUNZIONI HELPER

function plotSVMHeatmap(resultsTbl, outputPath, plotTitle)
% Genera una heatmap 2D per i parametri SVM

    % Si estraggono i valori unici testati per BoxConstraint (C) e KernelScale
    CValues = unique(resultsTbl.BoxConstraint);
    KSValues = unique(resultsTbl.KernelScale);

    % Si prealloca una matrice Z riempita di NaN per ospitare i valori di F1-score
    Z = NaN(numel(KSValues), numel(CValues));

    % Scorre la tabella dei risultati per posizionare ciascun F1-score 
    % nella cella corrispondente della griglia
    for r = 1:height(resultsTbl)
        i = find(KSValues == resultsTbl.KernelScale(r));
        j = find(CValues == resultsTbl.BoxConstraint(r));
        Z(i, j) = resultsTbl.F1(r);
    end

    % Generazione heatmap
    f = figure('Visible', 'off', 'Color', 'w');
    imagesc(CValues, KSValues, Z);
    colorbar;
    xlabel('BoxConstraint C');
    ylabel('KernelScale');
    title(plotTitle);

    % Si forzano gli assi a mostrare i valori esatti testati
    set(gca, 'XTick', CValues, 'YTick', KSValues);

    % Si sovrascrive il valore esatto di F1-score in formato testuale sopra 
    % ogni singola cella per facilitarne la lettura
    for i = 1:numel(KSValues)
        for j = 1:numel(CValues)
            text(CValues(j), KSValues(i), sprintf('%.3f', Z(i,j)), ...
                'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
        end
    end

    % Salva immagine su disco
    saveas(f, outputPath);
    close(f);
end

function plotRFLinePlot(resultsTbl, outputPath, plotTitle)
% Genera un grafico a linee per i parametri Random Forest.

    % Si estraggono i valori unici per il numero massimo di split testati
    maxSplitsValues = unique(resultsTbl.MaxNumSplits);

    f = figure('Visible', 'off', 'Color', 'w');
    hold on;

    % Filtra la tabella, ordina i dati per numero di cicli 
    % (alberi) e si traccia la curva prestazionale
    for i = 1:numel(maxSplitsValues)
        ms = maxSplitsValues(i);
        subset = resultsTbl(resultsTbl.MaxNumSplits == ms, :);
        subset = sortrows(subset, 'NumLearningCycles');
        plot(subset.NumLearningCycles, subset.F1, '-o', 'LineWidth', 1.8, ...
            'DisplayName', sprintf('MaxNumSplits = %d', ms));
    end

    % Formattazione del grafico con etichette, titolo, legenda e griglia
    xlabel('NumLearningCycles');
    ylabel('F1-score');
    title(plotTitle);
    legend('Location', 'best');
    grid on;

    % Salvataggio dell'immagine finale e pulizia della memoria
    saveas(f, outputPath);
    close(f);
end
