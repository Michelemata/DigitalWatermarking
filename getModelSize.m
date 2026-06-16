function sizeMB = getModelSize(modelObj, baseName)
% Stima la dimensione di un modello salvandolo temporaneamente.

    % Se il nome del file non viene specificato in input, si assegna un nome 
    % temporaneo di default per evitare errori.
    if nargin < 2 || isempty(baseName)
        baseName = 'temp_model';
    end

    % Si costruisce il percorso completo del file sfruttando la directory 
    % temporanea di sistema 'tempdir', per non inquinare la cartella di lavoro
    tempFile = fullfile(tempdir, [char(baseName), '.mat']);

    try
        % Assegna il modello a una variabile locale e la salva su disco
        modelToSave = modelObj; %#ok<NASGU>
        save(tempFile, 'modelToSave');

        % Interroga le proprietà del file appena creato
        info = dir(tempFile);

        % Converte la dimensione da Byte a Megabyte
        sizeMB = info.bytes / (1024^2);

        % Eimina il file temporaneo per liberare spazio su disco
        delete(tempFile);
    catch ME
        warning('Impossibile stimare la dimensione del modello %s: %s', baseName, ME.message);
        sizeMB = NaN;
    end
end
