function f=kmean_cosine(train,test,error,k)
    
    [idx_eucli,C_eucli]=kmeans(train(:,2:end),k, 'MaxIter',1000,'Distance','correlation','Start','cluster');
    centroids=zeros(k,14);
    for m=1:k
        centroids(m,:)=[mode(train((idx_eucli==m),1)),C_eucli(m,:)];
    end


    predicted=zeros(60,1);
    for i = 1:60
        dist_eucli=zeros(1,k);
        for j =1:k
            eucli=pdist([test(i,2:end);centroids(j,2:end)],'correlation');
            dist_eucli(1,j)=eucli;
        end

        [x,min_idx]=min(dist_eucli);
        predicted(i,1)=centroids(min_idx,1);
     end


    error=(60-sum(predicted==test(:,1)))/60;

    f=error;
end