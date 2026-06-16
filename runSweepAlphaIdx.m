%% SWEEP AUTOMATICO idx / alpha PER PIPELINE CLEAN
% Questo script esegue automaticamente modelli_ml.m per tutte le combinazioni
% di idx e alpha, senza modificare manualmente il main.
%
% Output principale:
% Results/clean/sweep_alpha_idx/all_trials_metrics_long.csv
% Results/clean/sweep_alpha_idx/summary_parametric_plots/*.png

clear; clc; close all;

% Valori sperimentali da analizzare
idxValues = [0.50, 0.25];
alphaValues = [0.02, 0.06];

allTrialsMetrics = table();
trialNumber = 0;

for iIdx = 1:numel(idxValues)
    for iAlpha = 1:numel(alphaValues)

        trialNumber = trialNumber + 1;

        % Variabili lette da modelli_ml.m
        sweepMode = true;
        sweepIdxValue = idxValues(iIdx);
        sweepAlphaValue = alphaValues(iAlpha);

        fprintf('\n============================================================\n');
        fprintf('Trial %d | idx = %.2f | alpha = %.3f\n', trialNumber, sweepIdxValue, sweepAlphaValue);
        fprintf('============================================================\n');

        % Esegue il main senza modificarne manualmente i parametri
        run('modelli_ml.m');

        % Recupera la tabella long prodotta dal trial corrente
        currentTrialPath = fullfile(figuresDir, 'trial_metrics_long.csv');
        if isfile(currentTrialPath)
            T = readtable(currentTrialPath);
            allTrialsMetrics = [allTrialsMetrics; T]; %#ok<AGROW>
        else
            warning('File trial_metrics_long.csv non trovato per il trial %d: %s', ...
                trialNumber, currentTrialPath);
        end

        close all;
    end
end

% Cartella finale per tabella aggregata e grafici riassuntivi
summaryDir = fullfile('Results', 'clean', 'sweep_alpha_idx', 'summary_parametric_plots');
if ~exist(summaryDir, 'dir')
    mkdir(summaryDir);
end

allMetricsPath = fullfile(summaryDir, 'all_trials_metrics_long.csv');
writetable(allTrialsMetrics, allMetricsPath);

fprintf('\nTabella aggregata salvata in:\n%s\n', allMetricsPath);
fprintf('\nSweep completato. Grafici salvati in:\n%s\n', summaryDir);
