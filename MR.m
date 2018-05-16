function [L] = MR(dataset,k,t)

    [m,n]=size(dataset);
    W = zeros(m, m);
    D = zeros(m, m);

    for i=1:m
        k_index=KNN(dataset(i,:),dataset,k);
        [ki,kj]=size(k_index);
        for j =1:ki
            sqDiffVector = dataset(i,:)-dataset(k_index(j),:);
            sqDiffVector = sqDiffVector.^2;
            sqDistances = sum(sqDiffVector);
            W(i, k_index(j)) = exp(-sqDistances / t);
            D(i, i) = D(i, i)+ W(i, k_index(j));           
        end
    end
    L = D - W 
end



function relustLabel = KNN(inx,data,k)
    [datarow , datacol] = size(data);
    diffMat = repmat(inx,[datarow,1]) - data ;
    distanceMat = sqrt(sum(diffMat.^2,2));
    [B , IX] = sort(distanceMat,'ascend');
    relustLabel = IX(1:k);

end
