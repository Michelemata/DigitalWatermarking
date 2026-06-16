function metriche = metrics(Ytrue, Ypred)

    % Si forzano gli input a essere vettori colonna di tipo double
    Ytrue = double(Ytrue(:));
    Ypred = double(Ypred(:));

    % Calcolo della matrice di confusione
    cm = confusionmat(Ytrue, Ypred, 'Order', [0 1]);

    % Estrazione dei singoli componenti dalla matrice di confusione
    TN = cm(1,1); FP = cm(1,2);
    FN = cm(2,1); TP = cm(2,2);

    % Calcolo delle metriche di classificazione
    acc  = (TP + TN) / max(sum(cm(:)), eps);
    prec = TP / max(TP + FP, eps);
    rec  = TP / max(TP + FN, eps);
    f1   = 2 * (prec * rec) / max(prec + rec, eps);

    % Aggrega i risultati in una singola struttura dati
    metriche = struct( ...
        'ConfusionMatrix', cm, ...
        'Accuracy', acc, ...
        'Precision', prec, ...
        'Recall', rec, ...
        'F1', f1);
end
