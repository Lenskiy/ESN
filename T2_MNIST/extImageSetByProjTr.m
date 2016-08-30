function [inputSetExt, outputSetExt] = extImageSetByProjTr(inputSet, outputSet, theta, phi, ksi)
    trNum = length(theta) * length(phi) * length(ksi);
    inputSetExt = zeros(size(inputSet,1), size(inputSet,2) * trNum);
    outputSetExt = zeros(size(outputSet,1), size(outputSet,2) * trNum);
    trainSize = size(inputSet,2);
    counter = 1;
    for j = 1:size(inputSet,2)
        j/trainSize
        img = reshape(inputSet(:,j), [28, 28]);
        trImgs = uint8(generateProjections(img, theta, phi, ksi));
        for k = 1:size(trImgs,3)
            img = trImgs(:,:,k);
            img = img(:);
            img(isnan(img)) = 0;
            inputSetExt(:, counter) = img;
            outputSetExt(:, counter) = outputSet(:,j);
            counter = counter + 1;
        end
    end
end