function createComplexityBarPlot(complexitySummary, figuresDir)

    % Se la directory di salvataggio non viene specificata,
    % si utilizza la cartella di lavoro corrente (pwd)
    if nargin < 2 || isempty(figuresDir)
        figuresDir = pwd;
    end

    % Si estraggono i nomi delle pipeline dalla tabella e si convertono 
    % in variabili categoriali per mantenere l'ordine corretto sull'asse x
    labels = categorical(complexitySummary.Pipeline);
    labels = reordercats(labels, complexitySummary.Pipeline);

    % Generazione del primo grafico: tempo medio di esecuzione per immagine
    f = figure('Name', 'Average pipeline time per image', 'Position', [100 100 1200 650]);
    bar(labels, complexitySummary.AvgPipelineTimePerImage_ms);
    ylabel('Average time per image [ms]');
    title('Computational complexity - Average pipeline time');
    grid on;
    xtickangle(25);
    saveas(f, fullfile(figuresDir, 'computational_complexity_avg_pipeline_time.png'));

    % Generazione del secondo grafico: tempo di addestramento dei modelli
    f = figure('Name', 'Training time by classifier', 'Position', [100 100 1200 650]);
    bar(labels, complexitySummary.TrainingTime_s);
    ylabel('Training time [s]');
    title('Computational complexity - Training time');
    grid on;
    xtickangle(25);
    saveas(f, fullfile(figuresDir, 'computational_complexity_training_time.png'));

    % Generazione del terzo grafico: dimensione su disco dei modelli salvati
    f = figure('Name', 'Model size by classifier', 'Position', [100 100 1200 650]);
    bar(labels, complexitySummary.ModelSizeMB);
    ylabel('Model size [MB]');
    title('Computational complexity - Model size');
    grid on;
    xtickangle(25);
    saveas(f, fullfile(figuresDir, 'computational_complexity_model_size.png'));
end
